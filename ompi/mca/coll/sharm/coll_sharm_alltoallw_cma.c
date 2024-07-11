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
 * @brief shared-memory based algorithm for Alltoallw using CMA ZeroCopy.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_alltoallw_cma(const void *sbuf, const int *scounts,
                        const int *sdispls,
                        struct ompi_datatype_t *const *sdtypes, void *rbuf,
                        const int *rcounts, const int *rdispls,
                        struct ompi_datatype_t *const *rdtypes,
                        struct ompi_communicator_t *comm,
                        mca_coll_base_module_t *module)
{
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;
    sharm_local_collectivies_data_t *coll_info = &(
        sharm_module->local_collectivies_info);

    int node_comm_rank = ompi_comm_rank(comm);
    int node_comm_size = ompi_comm_size(comm);

    const char *_sbuf = (char *) sbuf;
    const int *_scounts = scounts;
    struct ompi_datatype_t *const *_sdtypes = sdtypes;
    const int *_sdispls = sdispls;

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:alltoallw_cma: (%d/%d/%s)",
                         SHARM_COLL(alltoallw, sharm_module), node_comm_rank,
                         node_comm_size, comm->c_name));

    if (MPI_IN_PLACE == sbuf) {
        _scounts = rcounts;
        _sbuf = (char *) rbuf;
        _sdtypes = rdtypes;
        _sdispls = rdispls;
    }

    void *memory_map = sharm_module->local_op_memory_map;
    size_t *collectivies_info_bytes_received_by_rank = (size_t *) memory_map;

    coll_info->sdtypes_contiguous[node_comm_rank][0] = 1;
    for (int i = 0; i < node_comm_size; ++i) {
        coll_info->sdtypes_contiguous[node_comm_rank][0]
            &= ompi_datatype_is_contiguous_memory_layout(_sdtypes[i],
                                                         _scounts[i])
               & ompi_datatype_is_contiguous_memory_layout(rdtypes[i],
                                                           rcounts[i]);
        coll_info->sbuf[node_comm_rank][i] = (ptrdiff_t) (_sbuf + _sdispls[i]);
    }

    /*
     * Exchange collectivies info.
     */
    memset(collectivies_info_bytes_received_by_rank, 0,
           node_comm_size * sizeof(size_t));
    size_t collectivies_info_bytes_sended = 0;
    size_t collectivies_info_bytes_received = 0;
    size_t collectivies_info_bytes_total = (node_comm_size - 1)
                                           * coll_info->one_rank_block_size;
    while (collectivies_info_bytes_sended < coll_info->one_rank_block_size
           || collectivies_info_bytes_received
                  < collectivies_info_bytes_total) {
        for (int i = 0; i < node_comm_size; ++i) {
            if (i == node_comm_rank
                || collectivies_info_bytes_received_by_rank[i]
                       >= coll_info->one_rank_block_size) {
                continue;
            }
            int pop = sharm_queue_pop_contiguous(
                RESOLVE_COLLECTIVIES_DATA(sharm_module, i)
                    + collectivies_info_bytes_received_by_rank[i],
                i, comm, sharm_module);
            collectivies_info_bytes_received_by_rank[i] += pop;
            collectivies_info_bytes_received += pop;
        }

        if (collectivies_info_bytes_sended >= coll_info->one_rank_block_size) {
            continue;
        }

        int bytes_to_send = min(coll_info->one_rank_block_size
                                    - collectivies_info_bytes_sended,
                                shm_data->mu_queue_fragment_size);
        SHARM_PROFILING_TIME_START(sharm_module, alltoallw, push);
        int push = sharm_queue_push_contiguous(
            RESOLVE_COLLECTIVIES_DATA(sharm_module, node_comm_rank)
                + collectivies_info_bytes_sended,
            bytes_to_send, node_comm_rank, -1, comm, sharm_module);
        SHARM_PROFILING_TIME_STOP(sharm_module, alltoallw, push);
        collectivies_info_bytes_sended += push;
    }

    uint8_t is_contiguous_dtype = 1;
    for (int i = 0; i < node_comm_size; ++i) {
        is_contiguous_dtype &= coll_info->sdtypes_contiguous[i][0];
    }

    if (0 == is_contiguous_dtype) {
        opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                            "coll:sharm:%d:alltoallw_cma: (%d/%d/%s) "
                            "Unsupported non-contigous datatype",
                            SHARM_COLL(alltoallw, sharm_module), node_comm_rank,
                            node_comm_size, comm->c_name);
        // TODO: Fallback to cico
        return OMPI_ERR_NOT_SUPPORTED;
    }

    /*
     * Data exchange.
     */

    if (MPI_IN_PLACE != sbuf) {
        ompi_datatype_sndrcv(_sbuf + _sdispls[node_comm_rank],
                             _scounts[node_comm_rank], _sdtypes[node_comm_rank],
                             ((char *) rbuf) + rdispls[node_comm_rank],
                             rcounts[node_comm_rank], rdtypes[node_comm_rank]);
    }
    for (int i = 1; i < node_comm_size; ++i) {
        size_t rdtype_size;
        int recvfrom = (node_comm_rank + node_comm_size - i) % node_comm_size;
        ompi_datatype_type_size(rdtypes[recvfrom], &rdtype_size);
        int64_t total_size = rdtype_size * rcounts[recvfrom];
        char *rbuf_offset = (char *) rbuf + rdispls[recvfrom];
        int rc = sharm_cma_readv(SHARM_GET_RANK_PID(shm_data, recvfrom),
                                 rbuf_offset,
                                 (void *)
                                     coll_info->sbuf[recvfrom][node_comm_rank],
                                 total_size);
        if (rc != total_size) {
            return OMPI_ERROR;
        }
    }

    opal_atomic_wmb();

    OPAL_OUTPUT_VERBOSE(
        (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
         "coll:sharm:%d:alltoallw_cma: (%d/%d/%s), alltoallw complete",
         SHARM_COLL(alltoallw, sharm_module), node_comm_rank, node_comm_size,
         comm->c_name));

    return sharm_barrier_intra(comm, module);
}
