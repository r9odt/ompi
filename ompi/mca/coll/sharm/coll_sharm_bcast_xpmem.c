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
 * @brief shared-memory based algorithm for Bcast using XPMEM ZeroCopy.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_bcast_xpmem(void *buff, int count, ompi_datatype_t *datatype,
                      int root, ompi_communicator_t *comm,
                      mca_coll_base_module_t *module)
{
#if !(SHARM_CHECK_XPMEM_SUPPORT)
    return OMPI_ERR_NOT_AVAILABLE;
#else
    size_t ddt_size = 0;
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;
    sharm_local_collectivies_data_t *coll_info = &(
        sharm_module->local_collectivies_info);

    int node_comm_rank = ompi_comm_rank(comm);
    int node_comm_size = ompi_comm_size(comm);

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:bcast_xpmem: (%d/%d/%s) root %d",
                         SHARM_COLL(bcast, sharm_module), node_comm_rank,
                         node_comm_size, comm->c_name, root));

    void *memory_map = sharm_module->local_op_memory_map;
    size_t *collectivies_info_bytes_received_by_rank = (size_t *) memory_map;

    ompi_datatype_type_size(datatype, &ddt_size);
    coll_info->sdtypes_contiguous[node_comm_rank][0]
        = ompi_datatype_is_contiguous_memory_layout(datatype, count);

    size_t total_size = ddt_size * count;

    coll_info->sbuf[node_comm_rank][0] = (ptrdiff_t) buff;

    /*
     * Exchange collectivies info.
     */

    SHARM_PROFILING_TIME_START(sharm_module, bcast, collective_exchange);
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
            SHARM_PROFILING_TIME_START(sharm_module, bcast, pop);
            int pop = sharm_queue_pop_contiguous(
                RESOLVE_COLLECTIVIES_DATA(sharm_module, i)
                    + collectivies_info_bytes_received_by_rank[i],
                i, comm, sharm_module);
            SHARM_PROFILING_TIME_STOP(sharm_module, bcast, pop);
            collectivies_info_bytes_received_by_rank[i] += pop;
            collectivies_info_bytes_received += pop;
        }

        if (collectivies_info_bytes_sended >= coll_info->one_rank_block_size) {
            continue;
        }

        int bytes_to_send = min(coll_info->one_rank_block_size
                                    - collectivies_info_bytes_sended,
                                shm_data->mu_queue_fragment_size);
        SHARM_PROFILING_TIME_START(sharm_module, bcast, push);
        int push = sharm_queue_push_contiguous(
            RESOLVE_COLLECTIVIES_DATA(sharm_module, node_comm_rank)
                + collectivies_info_bytes_sended,
            bytes_to_send, node_comm_rank, -1, comm, sharm_module);
        SHARM_PROFILING_TIME_STOP(sharm_module, bcast, push);
        collectivies_info_bytes_sended += push;
    }

    uint8_t is_contiguous_dtype = 1;
    for (int i = 0; i < node_comm_size; ++i) {
        is_contiguous_dtype &= coll_info->sdtypes_contiguous[i][0];
    }

    SHARM_PROFILING_TIME_STOP(sharm_module, bcast, collective_exchange);

    if (0 == is_contiguous_dtype) {
        opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                            "coll:sharm:%d:bcast_xpmem: (%d/%d/%s) "
                            "Unsupported non-contigous datatype",
                            SHARM_COLL(bcast, sharm_module), node_comm_rank,
                            node_comm_size, comm->c_name);
        return OMPI_ERR_NOT_SUPPORTED;
    }

    if (root != node_comm_rank) {
        xpmem_segid_t segid = *((xpmem_segid_t *) shm_data->xpmem_segid[root]);
        xpmem_apid_t apid = -1;

        char *peer_buff = (char *) coll_info->sbuf[root][0];
        char *aligned_buff = (char *) (((uintptr_t) peer_buff)
                                       & ~(shm_data->mu_page_size - 1));
        size_t align_offset = peer_buff - aligned_buff;
        size_t aligned_total_size = (total_size + align_offset
                                     + shm_data->mu_page_size - 1)
                                    & ~(shm_data->mu_page_size - 1);
        SHARM_PROFILING_TIME_STOP(sharm_module, bcast, xpmem_attach);
        apid = xpmem_get(segid, XPMEM_RDWR, XPMEM_PERMIT_MODE, NULL);
        if (apid < 0) {
            opal_output_verbose(
                SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                "coll:sharm:%d:bcast_xpmem: (%d/%d/%s) can not get apid of"
                "shared memory region, error code %ld",
                SHARM_COLL(bcast, sharm_module), node_comm_rank, node_comm_size,
                comm->c_name, apid);
            return OMPI_ERROR;
        }

        struct xpmem_addr addr = {.apid = apid, .offset = (off_t) aligned_buff};
        int64_t xpmem_seg_addr = (int64_t) xpmem_attach(addr,
                                                        aligned_total_size,
                                                        (void *) NULL);

        if (xpmem_seg_addr < -1) {
            opal_output_verbose(
                SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                "coll:sharm:%d:bcast_xpmem: (%d/%d/%s) can not attach of"
                "shared memory region, error code %ld",
                SHARM_COLL(bcast, sharm_module), node_comm_rank, node_comm_size,
                comm->c_name, xpmem_seg_addr);
            xpmem_release(apid);
            return OMPI_ERROR;
        }
        SHARM_PROFILING_TIME_STOP(sharm_module, bcast, xpmem_attach);

        /* Copy */

        SHARM_PROFILING_TIME_START(sharm_module, bcast, copy);
        memcpy(buff, (void *) (xpmem_seg_addr + align_offset), total_size);
        SHARM_PROFILING_TIME_STOP(sharm_module, bcast, copy);

        opal_atomic_wmb();

        xpmem_detach((void *) xpmem_seg_addr);
        xpmem_release(apid);
    }

    SHARM_PROFILING_TIME_START(sharm_module, bcast, zcopy_barrier);
    sharm_barrier_sense_reversing(comm, module);
    SHARM_PROFILING_TIME_STOP(sharm_module, bcast, zcopy_barrier);

    OPAL_OUTPUT_VERBOSE(
        (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
         "coll:sharm:%d:bcast_xpmem: (%d/%d/%s), root %d bcast complete",
         SHARM_COLL(bcast, sharm_module), node_comm_rank, node_comm_size,
         comm->c_name, root));

    return OMPI_SUCCESS;
#endif
}
