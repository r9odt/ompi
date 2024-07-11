/** @file */

#include "coll_sharm.h"

/*
 * External variables.
 */

extern int mca_coll_sharm_stream;
/*
 * Local functions.
 */

/**
 * @brief shared-memory based algorithm for Bcast using CMA ZeroCopy.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_bcast_cma(void *buff, int count, ompi_datatype_t *datatype, int root,
                    ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    size_t ddt_size = 0;
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;

    int node_comm_rank = ompi_comm_rank(comm);
    int node_comm_size = ompi_comm_size(comm);

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:bcast_cma: (%d/%d/%s) root %d",
                         SHARM_COLL(bcast, sharm_module), node_comm_rank,
                         node_comm_size, comm->c_name, root));

    void *memory_map = sharm_module->local_op_memory_map;
    size_t *collectivies_info_bytes_received_by_rank = (size_t *) memory_map;

    ompi_datatype_type_size(datatype, &ddt_size);

    /*
     * Use collectivies exchange v2.1
     */

    char *my_coll_info_block = RESOLVE_COLLECTIVIES_DATA(sharm_module,
                                                         node_comm_rank);
    *my_coll_info_block = ompi_datatype_is_contiguous_memory_layout(datatype,
                                                                    count);
    *(ptrdiff_t *) (my_coll_info_block + sizeof(char)) = (ptrdiff_t) buff;

    OPAL_OUTPUT_VERBOSE(
        (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
         "coll:sharm:%d:bcast_cma: (%d/%d/%s), root %d bcast my buff is %p",
         SHARM_COLL(bcast, sharm_module), node_comm_rank, node_comm_size,
         comm->c_name, root, buff));

    /*
     * Exchange collectivies info.
     */

    SHARM_PROFILING_TIME_START(sharm_module, bcast, collective_exchange);
    memset(collectivies_info_bytes_received_by_rank, 0,
           node_comm_size * sizeof(size_t));
    size_t collectivies_info_bytes_sended = 0;
    size_t collectivies_info_bytes_received = 0;
    size_t collectivies_info_bytes_total_one_rank = sizeof(char)
                                                    + sizeof(ptrdiff_t);
    size_t collectivies_info_bytes_total
        = (node_comm_size - 1) * collectivies_info_bytes_total_one_rank;
    while (
        collectivies_info_bytes_sended < collectivies_info_bytes_total_one_rank
        || collectivies_info_bytes_received < collectivies_info_bytes_total) {
        for (int i = 0; i < node_comm_size; ++i) {
            if (i == node_comm_rank
                || collectivies_info_bytes_received_by_rank[i]
                       >= collectivies_info_bytes_total_one_rank) {
                continue;
            }
            SHARM_PROFILING_TIME_START(sharm_module, bcast, pop);
            int pop = sharm_queue_pop_contiguous(
                RESOLVE_COLLECTIVIES_DATA(sharm_module, i)
                    + collectivies_info_bytes_received_by_rank[i],
                i, comm, sharm_module);
            SHARM_PROFILING_TIME_STOP(sharm_module, bcast, pop);
            collectivies_info_bytes_received_by_rank[i] += pop;
            collectivies_info_bytes_received += pop;
        }

        if (collectivies_info_bytes_sended
            >= collectivies_info_bytes_total_one_rank) {
            continue;
        }

        int bytes_to_send = min(collectivies_info_bytes_total_one_rank
                                    - collectivies_info_bytes_sended,
                                shm_data->mu_queue_fragment_size);
        SHARM_PROFILING_TIME_START(sharm_module, bcast, push);
        int push = sharm_queue_push_contiguous(
            my_coll_info_block + collectivies_info_bytes_sended, bytes_to_send,
            node_comm_rank, -1, comm, sharm_module);
        SHARM_PROFILING_TIME_STOP(sharm_module, bcast, push);
        collectivies_info_bytes_sended += push;
    }

    uint8_t is_contiguous_dtype = 1;
    for (int i = 0; i < node_comm_size; ++i) {
        char *coll_info_is_contigous = RESOLVE_COLLECTIVIES_DATA(sharm_module,
                                                                 i);
        is_contiguous_dtype &= *coll_info_is_contigous;
    }

    SHARM_PROFILING_TIME_STOP(sharm_module, bcast, collective_exchange);

    if (0 == is_contiguous_dtype) {
        opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                            "coll:sharm:%d:bcast_cma: (%d/%d/%s) "
                            "Unsupported non-contigous datatype",
                            SHARM_COLL(bcast, sharm_module), node_comm_rank,
                            node_comm_size, comm->c_name);
        return OMPI_ERR_NOT_SUPPORTED;
    }

    int total_size = ddt_size * count;
    if (root != node_comm_rank) {
        ptrdiff_t *root_coll_info_sbuf
            = (ptrdiff_t *) (RESOLVE_COLLECTIVIES_DATA(sharm_module, root)
                             + sizeof(char));
        OPAL_OUTPUT_VERBOSE(
            (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
             "coll:sharm:%d:bcast_cma: (%d/%d/%s), root %d bcast read from %p",
             SHARM_COLL(bcast, sharm_module), node_comm_rank, node_comm_size,
             comm->c_name, root, (void *) *root_coll_info_sbuf));
        SHARM_PROFILING_TIME_START(sharm_module, bcast, copy);
        int rc = sharm_cma_readv(SHARM_GET_RANK_PID(shm_data, root), buff,
                                 (void *) (*root_coll_info_sbuf), total_size);
        SHARM_PROFILING_TIME_STOP(sharm_module, bcast, copy);

        if (rc != total_size) {
            return OMPI_ERROR;
        }
    }

    opal_atomic_wmb();

    OPAL_OUTPUT_VERBOSE(
        (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
         "coll:sharm:%d:bcast_cma: (%d/%d/%s), root %d bcast complete",
         SHARM_COLL(bcast, sharm_module), node_comm_rank, node_comm_size,
         comm->c_name, root));

    SHARM_PROFILING_TIME_START(sharm_module, bcast, zcopy_barrier);
    int err = sharm_barrier_gather_cico(root, comm, module);
    SHARM_PROFILING_TIME_STOP(sharm_module, bcast, zcopy_barrier);
    // return sharm_barrier_sense_reversing(comm, module);
    return err;
}
