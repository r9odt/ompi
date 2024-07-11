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
 * @brief shared-memory based algorithm for Allgatherv using CMA ZeroCopy.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_allgatherv_cma(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                         void *rbuf, const int *rcounts, const int *displs,
                         ompi_datatype_t *rdtype, ompi_communicator_t *comm,
                         mca_coll_base_module_t *module)
{
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;
    sharm_local_collectivies_data_t *coll_info = &(
        sharm_module->local_collectivies_info);

    int node_comm_rank = ompi_comm_rank(comm);
    int node_comm_size = ompi_comm_size(comm);

    const char *_sbuf = sbuf;
    int _scount = scount;
    ompi_datatype_t *_sdtype = sdtype;

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:allgatherv_cma: (%d/%d/%s)",
                         SHARM_COLL(allgatherv, sharm_module), node_comm_rank,
                         node_comm_size, comm->c_name));

    size_t sdtype_size = 0;
    size_t rdtype_size = 0;
    ompi_datatype_type_size(_sdtype, &sdtype_size);
    ompi_datatype_type_size(rdtype, &rdtype_size);

    ptrdiff_t sext;
    ompi_datatype_type_extent(_sdtype, &sext);
    ptrdiff_t rext;
    ompi_datatype_type_extent(rdtype, &rext);

    /*
     * If sbuf is MPI_IN_PLACE - send data from rbuf.
     */
    if (MPI_IN_PLACE == sbuf) {
        _scount = rcounts[node_comm_rank];
        _sbuf = (char *) rbuf + rext * displs[node_comm_rank];
        _sdtype = rdtype;
    }

    coll_info->sdtypes_ext[node_comm_rank][0] = sext;
    coll_info->rdtypes_ext[node_comm_rank][0] = rext;
    coll_info->sbuf[node_comm_rank][0] = (ptrdiff_t) _sbuf;
    coll_info->sdtypes_contiguous[node_comm_rank][0] = 1;
    for (int i = 0; i < node_comm_size; ++i) {
        coll_info->sdtypes_contiguous[node_comm_rank][0]
            &= ompi_datatype_is_contiguous_memory_layout(_sdtype, _scount)
               & ompi_datatype_is_contiguous_memory_layout(rdtype, rcounts[i]);
    }

    void *memory_map = sharm_module->local_op_memory_map;
    size_t *collectivies_info_bytes_received_by_rank = (size_t *) memory_map;

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
        SHARM_PROFILING_TIME_START(sharm_module, allgatherv, push);
        int push = sharm_queue_push_contiguous(
            RESOLVE_COLLECTIVIES_DATA(sharm_module, node_comm_rank)
                + collectivies_info_bytes_sended,
            bytes_to_send, node_comm_rank, -1, comm, sharm_module);
        SHARM_PROFILING_TIME_STOP(sharm_module, allgatherv, push);
        collectivies_info_bytes_sended += push;
    }

    uint8_t is_contiguous_dtype = 1;
    for (int i = 0; i < node_comm_size; ++i) {
        is_contiguous_dtype &= coll_info->sdtypes_contiguous[i][0];
    }

    if (0 == is_contiguous_dtype) {
        opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                            "coll:sharm:%d:allgatherv_cma: (%d/%d/%s) "
                            "Unsupported non-contigous datatype",
                            SHARM_COLL(allgatherv, sharm_module),
                            node_comm_rank, node_comm_size, comm->c_name);
        // TODO: Fallback to cico
        return OMPI_ERR_NOT_SUPPORTED;
    }

    /*
     * Data exchange.
     */

    for (int i = 0; i < node_comm_size; ++i) {
        int64_t total_size = rdtype_size * rcounts[i];
        char *rbuf_offset = (char *) rbuf + displs[i] * rext;

        if (OPAL_UNLIKELY(i == node_comm_rank && MPI_IN_PLACE != sbuf)) {
            ompi_datatype_sndrcv((char *) _sbuf, _scount, _sdtype, rbuf_offset,
                                 rcounts[i], rdtype);
            continue;
        }
        
        int rc = sharm_cma_readv(SHARM_GET_RANK_PID(shm_data, i), rbuf_offset,
                                 (void *) coll_info->sbuf[i][0], total_size);
        if (rc != total_size) {
            return OMPI_ERROR;
        }
    }

    opal_atomic_wmb();

    OPAL_OUTPUT_VERBOSE(
        (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
         "coll:sharm:%d:allgatherv_cma: (%d/%d/%s), allgatherv complete",
         SHARM_COLL(allgatherv, sharm_module), node_comm_rank, node_comm_size,
         comm->c_name));

    return sharm_barrier_intra(comm, module);
}
