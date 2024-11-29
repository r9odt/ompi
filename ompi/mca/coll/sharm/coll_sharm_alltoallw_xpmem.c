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
int sharm_alltoallw_xpmem(const void *sbuf, const int *scounts,
                          const int *sdispls,
                          struct ompi_datatype_t *const *sdtypes, void *rbuf,
                          const int *rcounts, const int *rdispls,
                          struct ompi_datatype_t *const *rdtypes,
                          struct ompi_communicator_t *comm,
                          mca_coll_base_module_t *module)
{
#if !(SHARM_CHECK_XPMEM_SUPPORT)
    return OMPI_ERR_NOT_AVAILABLE;
#else
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;
    sharm_local_collectivies_data_t *coll_info = &(
        sharm_module->local_collectivies_info);

    int comm_rank = ompi_comm_rank(comm);
    int comm_size = ompi_comm_size(comm);

    const char *_sbuf = (char *) sbuf;
    const int *_scounts = scounts;
    struct ompi_datatype_t *const *_sdtypes = sdtypes;
    const int *_sdispls = sdispls;

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:alltoallw_xpmem: (%d/%d/%s)",
                         SHARM_COLL(alltoallw, sharm_module), comm_rank,
                         comm_size, comm->c_name));

    if (MPI_IN_PLACE == sbuf) {
        _scounts = rcounts;
        _sbuf = (char *) rbuf;
        _sdtypes = rdtypes;
        _sdispls = rdispls;
    }

    void *memory_map = sharm_module->local_op_memory_map;
    size_t *collectivies_info_bytes_received_by_rank = (size_t *) memory_map;

    coll_info->sdtypes_contiguous[comm_rank][0] = 1;
    for (int i = 0; i < comm_size; ++i) {
        coll_info->sdtypes_contiguous[comm_rank][0]
            &= ompi_datatype_is_contiguous_memory_layout(_sdtypes[i],
                                                         _scounts[i])
               & ompi_datatype_is_contiguous_memory_layout(rdtypes[i],
                                                           rcounts[i]);
        coll_info->sbuf[comm_rank][i] = (ptrdiff_t) (_sbuf + _sdispls[i]);
    }

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
        SHARM_PROFILING_TIME_START(sharm_module, alltoallw, push);
        int push = sharm_queue_push_contiguous(
            RESOLVE_COLLECTIVIES_DATA(sharm_module, comm_rank)
                + collectivies_info_bytes_sended,
            bytes_to_send, comm_rank, -1, comm, sharm_module);
        SHARM_PROFILING_TIME_STOP(sharm_module, alltoallw, push);
        collectivies_info_bytes_sended += push;
    }

    uint8_t is_contiguous_dtype = 1;
    for (int i = 0; i < comm_size; ++i) {
        is_contiguous_dtype &= coll_info->sdtypes_contiguous[i][0];
    }

    if (0 == is_contiguous_dtype) {
        opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                            "coll:sharm:%d:alltoallw_xpmem: (%d/%d/%s) "
                            "Unsupported non-contigous datatype",
                            SHARM_COLL(alltoallw, sharm_module), comm_rank,
                            comm_size, comm->c_name);
        // TODO: Fallback to cico
        return OMPI_ERR_NOT_SUPPORTED;
    }

    /*
     * Data exchange.
     */

    if (MPI_IN_PLACE != sbuf) {
        ompi_datatype_sndrcv(_sbuf + _sdispls[comm_rank],
                             _scounts[comm_rank], _sdtypes[comm_rank],
                             ((char *) rbuf) + rdispls[comm_rank],
                             rcounts[comm_rank], rdtypes[comm_rank]);
    }
    for (int i = 1; i < comm_size; ++i) {
        size_t rdtype_size;
        int recvfrom = (comm_rank + comm_size - i) % comm_size;
        ompi_datatype_type_size(rdtypes[recvfrom], &rdtype_size);
        int64_t total_size = rdtype_size * rcounts[recvfrom];
        char *rbuf_offset = (char *) rbuf + rdispls[recvfrom];

        xpmem_segid_t segid = *((xpmem_segid_t *) shm_data->xpmem_segid[recvfrom]);
        xpmem_apid_t apid = -1;

        char *peer_buff = (char *) coll_info->sbuf[recvfrom][comm_rank];
        char *aligned_buff = (char *) (((uintptr_t) peer_buff)
                                       & ~(shm_data->mu_page_size - 1));
        size_t align_offset = peer_buff - aligned_buff;
        size_t aligned_total_size = (total_size + align_offset
                                     + shm_data->mu_page_size - 1)
                                    & ~(shm_data->mu_page_size - 1);
        SHARM_PROFILING_TIME_START(sharm_module, alltoallw, xpmem_attach);
        apid = xpmem_get(segid, XPMEM_RDWR, XPMEM_PERMIT_MODE, NULL);
        if (apid < 0) {
            opal_output_verbose(
                SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                "coll:sharm:%d:alltoallw_xpmem: (%d/%d/%s) can not get apid of"
                "shared memory region, error code %ld",
                SHARM_COLL(alltoallw, sharm_module), comm_rank,
                comm_size, comm->c_name, apid);
            return OMPI_ERROR;
        }

        struct xpmem_addr addr = {.apid = apid, .offset = (off_t) aligned_buff};
        int64_t xpmem_seg_addr = (int64_t) xpmem_attach(addr,
                                                        aligned_total_size,
                                                        (void *) NULL);

        if (xpmem_seg_addr < -1) {
            opal_output_verbose(
                SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                "coll:sharm:%d:alltoallw_xpmem: (%d/%d/%s) can not attach of"
                "shared memory region, error code %ld",
                SHARM_COLL(alltoallw, sharm_module), comm_rank,
                comm_size, comm->c_name, xpmem_seg_addr);
            xpmem_release(apid);
            return OMPI_ERROR;
        }
        SHARM_PROFILING_TIME_STOP(sharm_module, alltoallw, xpmem_attach);

        SHARM_PROFILING_TIME_START(sharm_module, alltoallw, copy);
        memcpy(rbuf_offset, (void *) (xpmem_seg_addr + align_offset),
               total_size);
        SHARM_PROFILING_TIME_STOP(sharm_module, alltoallw, copy);

        opal_atomic_wmb();

        xpmem_detach((void *) xpmem_seg_addr);
        xpmem_release(apid);
    }

    opal_atomic_wmb();

    OPAL_OUTPUT_VERBOSE(
        (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
         "coll:sharm:%d:alltoallw_xpmem: (%d/%d/%s), alltoallw complete",
         SHARM_COLL(alltoallw, sharm_module), comm_rank, comm_size,
         comm->c_name));

    return sharm_barrier_intra(comm, module);
#endif
}
