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
 * @brief shared-memory based algorithm for Gatherv using CMA ZeroCopy.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_gatherv_cma(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                      void *rbuf, const int *rcounts, const int *displs,
                      ompi_datatype_t *rdtype, int root,
                      ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;
    sharm_local_collectivies_data_t *coll_info = &(
        sharm_module->local_collectivies_info);

    int comm_rank = ompi_comm_rank(comm);
    int comm_size = ompi_comm_size(comm);

    const char *_sbuf = sbuf;
    int _scount = scount;
    ompi_datatype_t *_sdtype = sdtype;

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:gatherv_cico: (%d/%d/%s) root %d",
                         SHARM_COLL(gatherv, sharm_module), comm_rank,
                         comm_size, comm->c_name, root));

    ptrdiff_t rext;
    ompi_datatype_type_extent(rdtype, &rext);

    if (root == comm_rank && MPI_IN_PLACE == sbuf) {
        _scount = rcounts[comm_rank];
        _sbuf = (char *) rbuf + rext * displs[comm_rank];
        _sdtype = rdtype;
    }

    coll_info->sdtypes_contiguous[comm_rank][0]
        = ompi_datatype_is_contiguous_memory_layout(_sdtype, _scount);
    if (root == comm_rank) {
        for (int i = 0; i < comm_size; ++i) {
            coll_info->rbuf[comm_rank][i]
                = (ptrdiff_t) ((char *) rbuf + rext * displs[i]);
            coll_info->sdtypes_contiguous[comm_rank][0]
                &= ompi_datatype_is_contiguous_memory_layout(rdtype,
                                                             rcounts[i]);
        }
    }

    void *memory_map = sharm_module->local_op_memory_map;
    size_t *collectivies_info_bytes_received_by_rank = (size_t *) memory_map;

    /*
     * Exchange collectivies info.
     */
    memset(collectivies_info_bytes_received_by_rank, 0,
           comm_size * sizeof(size_t));
    size_t collectivies_info_bytes_sended = 0;
    size_t collectivies_info_bytes_received = 0;
    size_t collectivies_info_bytes_total = (comm_size - 1)
                                           * coll_info->one_rank_block_size;
    while (collectivies_info_bytes_sended < coll_info->one_rank_block_size
           || collectivies_info_bytes_received
                  < collectivies_info_bytes_total) {
        for (int i = 0; i < comm_size; ++i) {
            if (i == comm_rank
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
        SHARM_PROFILING_TIME_START(sharm_module, gatherv, push);
        int push = sharm_queue_push_contiguous(
            RESOLVE_COLLECTIVIES_DATA(sharm_module, comm_rank)
                + collectivies_info_bytes_sended,
            bytes_to_send, comm_rank, -1, comm, sharm_module);
        SHARM_PROFILING_TIME_STOP(sharm_module, gatherv, push);
        collectivies_info_bytes_sended += push;
    }

    uint8_t is_contiguous_dtype = 1;
    for (int i = 0; i < comm_size; ++i) {
        is_contiguous_dtype &= coll_info->sdtypes_contiguous[i][0];
    }

    if (0 == is_contiguous_dtype) {
        opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                            "coll:sharm:%d:gatherv_cma: (%d/%d/%s) "
                            "Unsupported non-contigous datatype",
                            SHARM_COLL(gatherv, sharm_module), comm_rank,
                            comm_size, comm->c_name);
        SHARM_PROFILING_TOTAL_TIME_STOP(sharm_module, gatherv);
        // TODO: Fallback to cico
        return OMPI_ERR_NOT_SUPPORTED;
    }

    /*
     * Data exchange.
     */

    if (root == comm_rank && MPI_IN_PLACE != sbuf) {
        char *rbuf_offset = (char *) rbuf + rext * displs[comm_rank];
        ompi_datatype_sndrcv((char *) _sbuf, _scount, _sdtype, rbuf_offset,
                             rcounts[comm_rank], rdtype);
    } else {
        size_t sdtype_size;
        ompi_datatype_type_size(_sdtype, &sdtype_size);
        int64_t total_size = sdtype_size * _scount;
        int rc = sharm_cma_writev(SHARM_GET_RANK_PID(shm_data, root), _sbuf,
                                  (void *)
                                      coll_info->rbuf[root][comm_rank],
                                  total_size);
        if (rc != total_size) {
            return OMPI_ERROR;
        }
    }

    opal_atomic_wmb();

    sharm_barrier_bcast_cico(root, comm, module);

    OPAL_OUTPUT_VERBOSE(
        (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
         "coll:sharm:%d:gatherv_cma: (%d/%d/%s), root %d gatherv complete",
         SHARM_COLL(gatherv, sharm_module), comm_rank, comm_size,
         comm->c_name, root));

    return OMPI_SUCCESS;
}
