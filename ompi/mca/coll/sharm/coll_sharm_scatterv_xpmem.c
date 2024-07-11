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
 * @brief shared-memory based algorithm for Scatterv using XPMEM ZeroCopy.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_scatterv_xpmem(const void *sbuf, const int *scounts,
                         const int *displs, ompi_datatype_t *sdtype, void *rbuf,
                         int rcount, ompi_datatype_t *rdtype, int root,
                         ompi_communicator_t *comm,
                         mca_coll_base_module_t *module)
{
#if !(SHARM_CHECK_XPMEM_SUPPORT)
    return OMPI_ERR_NOT_AVAILABLE;
#else
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;
    sharm_local_collectivies_data_t *coll_info = &(
        sharm_module->local_collectivies_info);

    int node_comm_rank = ompi_comm_rank(comm);
    int node_comm_size = ompi_comm_size(comm);

    int _rcount = rcount;
    ompi_datatype_t *_rdtype = rdtype;
    char *_rbuf = rbuf;

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:scatterv_xpmem: (%d/%d/%s) root %d",
                         SHARM_COLL(scatterv, sharm_module), node_comm_rank,
                         node_comm_size, comm->c_name, root));

    ptrdiff_t sext;
    ompi_datatype_type_extent(sdtype, &sext);

    if (rbuf == MPI_IN_PLACE) {
        _rcount = scounts[node_comm_rank];
        _rbuf = (char *) sbuf + sext * displs[node_comm_rank];
        _rdtype = sdtype;
    }

    ptrdiff_t rext;
    ompi_datatype_type_extent(_rdtype, &rext);
    size_t rdtype_size;
    ompi_datatype_type_size(_rdtype, &rdtype_size);
    coll_info->sdtypes_ext[node_comm_rank][0] = sext;
    coll_info->rdtypes_ext[node_comm_rank][0] = rext;

    coll_info->sdtypes_contiguous[node_comm_rank][0]
        = ompi_datatype_is_contiguous_memory_layout(_rdtype,
                                                    _rcount * node_comm_size);
    if (root == node_comm_rank) {
        for (int i = 0; i < node_comm_size; ++i) {
            coll_info->sdtypes_contiguous[node_comm_rank][0]
                &= ompi_datatype_is_contiguous_memory_layout(
                    sdtype, scounts[i] * node_comm_size);
            coll_info->sbuf[node_comm_rank][i]
                = (ptrdiff_t) (((char *) sbuf) + sext * displs[i]);
        }
        if (rbuf != MPI_IN_PLACE) {
            ompi_datatype_sndrcv((char *) coll_info
                                     ->sbuf[node_comm_rank][node_comm_rank],
                                 scounts[node_comm_rank], sdtype, _rbuf,
                                 _rcount, _rdtype);
        }
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
        SHARM_PROFILING_TIME_START(sharm_module, scatterv, push);
        int push = sharm_queue_push_contiguous(
            RESOLVE_COLLECTIVIES_DATA(sharm_module, node_comm_rank)
                + collectivies_info_bytes_sended,
            bytes_to_send, node_comm_rank, -1, comm, sharm_module);
        SHARM_PROFILING_TIME_STOP(sharm_module, scatterv, push);
        collectivies_info_bytes_sended += push;
    }

    uint8_t is_contiguous_dtype = 1;
    for (int i = 0; i < node_comm_size; ++i) {
        is_contiguous_dtype &= coll_info->sdtypes_contiguous[i][0];
    }

    if (0 == is_contiguous_dtype) {
        opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                            "coll:sharm:%d:scatterv_xpmem: (%d/%d/%s) "
                            "Unsupported non-contigous datatype",
                            SHARM_COLL(scatterv, sharm_module), node_comm_rank,
                            node_comm_size, comm->c_name);
        // TODO: Fallback to cico
        return OMPI_ERR_NOT_SUPPORTED;
    }

    /*
     * Data exchange.
     */

    if (root != node_comm_rank) {
        xpmem_segid_t segid = *((xpmem_segid_t *) shm_data->xpmem_segid[root]);
        xpmem_apid_t apid = -1;
        int64_t total_size = rdtype_size * _rcount;

        char *peer_buff = (char *) coll_info->sbuf[root][node_comm_rank];
        char *aligned_buff = (char *) (((uintptr_t) peer_buff)
                                       & ~(shm_data->mu_page_size - 1));
        size_t align_offset = peer_buff - aligned_buff;
        size_t aligned_total_size = (total_size + align_offset
                                     + shm_data->mu_page_size - 1)
                                    & ~(shm_data->mu_page_size - 1);
        SHARM_PROFILING_TIME_START(sharm_module, scatterv, xpmem_attach);
        apid = xpmem_get(segid, XPMEM_RDWR, XPMEM_PERMIT_MODE, NULL);
        if (apid < 0) {
            opal_output_verbose(
                SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                "coll:sharm:%d:scatterv_xpmem: (%d/%d/%s) can not get apid of"
                "shared memory region, error code %ld",
                SHARM_COLL(scatterv, sharm_module), node_comm_rank,
                node_comm_size, comm->c_name, apid);
            return OMPI_ERROR;
        }

        struct xpmem_addr addr = {.apid = apid, .offset = (off_t) aligned_buff};
        int64_t xpmem_seg_addr = (int64_t) xpmem_attach(addr,
                                                        aligned_total_size,
                                                        (void *) NULL);

        if (xpmem_seg_addr < -1) {
            opal_output_verbose(
                SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                "coll:sharm:%d:scatterv_xpmem: (%d/%d/%s) can not attach of"
                "shared memory region, error code %ld",
                SHARM_COLL(scatterv, sharm_module), node_comm_rank,
                node_comm_size, comm->c_name, xpmem_seg_addr);
            xpmem_release(apid);
            return OMPI_ERROR;
        }
        SHARM_PROFILING_TIME_STOP(sharm_module, scatterv, xpmem_attach);

        /* Copy */

        SHARM_PROFILING_TIME_START(sharm_module, scatterv, copy);
        memcpy(_rbuf, (void *) (xpmem_seg_addr + align_offset), total_size);
        SHARM_PROFILING_TIME_STOP(sharm_module, scatterv, copy);

        opal_atomic_wmb();

        xpmem_detach((void *) xpmem_seg_addr);
        xpmem_release(apid);
    }

    SHARM_PROFILING_TIME_START(sharm_module, scatterv, zcopy_barrier);
    sharm_barrier_sense_reversing(comm, module);
    SHARM_PROFILING_TIME_STOP(sharm_module, scatterv, zcopy_barrier);

    OPAL_OUTPUT_VERBOSE(
        (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
         "coll:sharm:%d:scatterv_xpmem: (%d/%d/%s), root %d scatterv complete",
         SHARM_COLL(scatterv, sharm_module), node_comm_rank, node_comm_size,
         comm->c_name, root));

    return OMPI_SUCCESS;
#endif
}
