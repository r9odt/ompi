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
 * @brief shared-memory based algorithm for Reduce using CMA approach.
 * Use flat tree of processes.
 * Support non-commutative operations.
 */
int sharm_reduce_cma(const void *sbuf, void *rbuf, int count,
                     struct ompi_datatype_t *dtype, struct ompi_op_t *op,
                     int root, struct ompi_communicator_t *comm,
                     mca_coll_base_module_t *module)
{
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;

    int node_comm_rank = ompi_comm_rank(comm);
    int node_comm_size = ompi_comm_size(comm);

    const char *_sbuf = sbuf;
    char *_rbuf = rbuf;

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:reduce_cma: (%d/%d/%s) root %d",
                         SHARM_COLL(reduce, sharm_module), node_comm_rank,
                         node_comm_size, comm->c_name, root));

    size_t ddt_size;
    ptrdiff_t extent;
    ompi_datatype_type_size(dtype, &ddt_size);
    ompi_datatype_type_extent(dtype, &extent);

    /*
     * If sbuf is MPI_IN_PLACE - send data from rbuf.
     */
    if (MPI_IN_PLACE == sbuf) {
        _sbuf = rbuf;
    }

    /*
     * Use collectivies exchange v2.1
     */

    char *my_coll_info_block = RESOLVE_COLLECTIVIES_DATA(sharm_module,
                                                         node_comm_rank);
    char *coll_info = my_coll_info_block;
    *my_coll_info_block = ompi_datatype_is_contiguous_memory_layout(dtype,
                                                                    count);
    coll_info = coll_info + sizeof(char);
    *(ptrdiff_t *) (coll_info) = (ptrdiff_t) _sbuf;

    void *memory_map = sharm_module->local_op_memory_map;
    size_t *collectivies_info_bytes_received_by_rank = (size_t *) memory_map;

    /*
     * Exchange collectivies info.
     */

    SHARM_PROFILING_TIME_START(sharm_module, reduce, collective_exchange);
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
            SHARM_PROFILING_TIME_START(sharm_module, reduce, pop);
            int pop = sharm_queue_pop_contiguous(
                RESOLVE_COLLECTIVIES_DATA(sharm_module, i)
                    + collectivies_info_bytes_received_by_rank[i],
                i, comm, sharm_module);
            SHARM_PROFILING_TIME_STOP(sharm_module, reduce, pop);
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
        SHARM_PROFILING_TIME_START(sharm_module, reduce, push);
        int push = sharm_queue_push_contiguous(
            my_coll_info_block + collectivies_info_bytes_sended, bytes_to_send,
            node_comm_rank, -1, comm, sharm_module);
        SHARM_PROFILING_TIME_STOP(sharm_module, reduce, push);
        collectivies_info_bytes_sended += push;
    }

    uint8_t is_contiguous_dtype = 1;
    for (int i = 0; i < node_comm_size; ++i) {
        char *coll_info_is_contigous = RESOLVE_COLLECTIVIES_DATA(sharm_module,
                                                                 i);
        is_contiguous_dtype &= *coll_info_is_contigous;
    }

    SHARM_PROFILING_TIME_STOP(sharm_module, reduce, collective_exchange);

    if (0 == is_contiguous_dtype) {
        opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                            "coll:sharm:%d:reduce_cma: (%d/%d/%s) "
                            "Unsupported non-contigous datatype",
                            SHARM_COLL(reduce, sharm_module), node_comm_rank,
                            node_comm_size, comm->c_name);
        return OMPI_ERR_NOT_SUPPORTED;
    }

    /*
     * Data exchange.
     */

    if (root == node_comm_rank) {
        char *temp_buffer = (char *) memory_map;
        char *reduce_temp_buffer = (char *) temp_buffer
                                   + shm_data->mu_queue_fragment_size;
        int fragment_num = 0;
        int64_t total_counts = count;
        int64_t segment_ddt_count = shm_data->mu_queue_fragment_size / extent;
        while (total_counts > 0) {
            int64_t min_counts = min(total_counts, segment_ddt_count);
            for (int i = node_comm_size - 1; i >= 0; --i) {
                if (OPAL_UNLIKELY(node_comm_rank == i)) {
                    if (i == node_comm_size - 1) {
                        SHARM_PROFILING_TIME_START(sharm_module, reduce,
                                                   reduce_operation);
                        ompi_datatype_copy_content_same_ddt(
                            dtype, min_counts, reduce_temp_buffer,
                            _sbuf + fragment_num * extent * segment_ddt_count);
                        SHARM_PROFILING_TIME_STOP(sharm_module, reduce,
                                                  reduce_operation);
                        continue;
                    }

                    SHARM_PROFILING_TIME_START(sharm_module, reduce,
                                               reduce_operation);
                    ompi_op_reduce(op,
                                   _sbuf
                                       + fragment_num * extent
                                             * segment_ddt_count,
                                   reduce_temp_buffer, min_counts, dtype);
                    SHARM_PROFILING_TIME_STOP(sharm_module, reduce,
                                              reduce_operation);
                    continue;
                }

                ptrdiff_t *peer_buff
                    = (ptrdiff_t *) (RESOLVE_COLLECTIVIES_DATA(sharm_module, i)
                                     + sizeof(char));

                size_t bytes_to_copy = min_counts * extent;
                int rc = sharm_cma_readv(SHARM_GET_RANK_PID(shm_data, i),
                                         temp_buffer,
                                         (void *) (((char *) (*peer_buff))
                                                   + fragment_num * extent
                                                         * segment_ddt_count),
                                         bytes_to_copy);

                if (rc != bytes_to_copy) {
                    return OMPI_ERROR;
                }

                if (i == node_comm_size - 1) {
                    SHARM_PROFILING_TIME_START(sharm_module, reduce, copy);
                    ompi_datatype_copy_content_same_ddt(dtype, min_counts,
                                                        reduce_temp_buffer,
                                                        (void *) temp_buffer);
                    SHARM_PROFILING_TIME_STOP(sharm_module, reduce, copy);
                } else {
                    SHARM_PROFILING_TIME_START(sharm_module, reduce,
                                               reduce_operation);
                    ompi_op_reduce(op, (void *) temp_buffer, reduce_temp_buffer,
                                   min_counts, dtype);
                    SHARM_PROFILING_TIME_STOP(sharm_module, reduce,
                                              reduce_operation);
                }
            }

            SHARM_PROFILING_TIME_START(sharm_module, reduce, reduce_operation);
            ompi_datatype_copy_content_same_ddt(dtype, min_counts,
                                                _rbuf
                                                    + fragment_num * extent
                                                          * segment_ddt_count,
                                                reduce_temp_buffer);
            SHARM_PROFILING_TIME_STOP(sharm_module, reduce, reduce_operation);

            total_counts -= segment_ddt_count;
            fragment_num++;
        }
    }

    SHARM_PROFILING_TIME_START(sharm_module, reduce, zcopy_barrier);
    sharm_barrier_sense_reversing(comm, module);
    SHARM_PROFILING_TIME_STOP(sharm_module, reduce, zcopy_barrier);

    OPAL_OUTPUT_VERBOSE(
        (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
         "coll:sharm:%d:reduce_cma: (%d/%d/%s), root %d reduce complete",
         SHARM_COLL(reduce, sharm_module), node_comm_rank, node_comm_size,
         comm->c_name, root));

    return OMPI_SUCCESS;
}
