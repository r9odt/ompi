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
 * @brief shared-memory based algorithm for Bcast using KNEM ZeroCopy.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_bcast_knem(void *buff, int count, ompi_datatype_t *datatype, int root,
                     ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
#if SHARM_CHECK_KNEM_SUPPORT == SHARM_FALSE
    return OMPI_ERR_NOT_AVAILABLE;
#else
    size_t ddt_size = 0;
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;
    sharm_local_collectivies_data_t *coll_info = &(
        sharm_module->local_collectivies_info);

    int comm_rank = ompi_comm_rank(comm);
    int comm_size = ompi_comm_size(comm);

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:bcast_kmem: (%d/%d/%s) root %d",
                         SHARM_COLL(bcast, sharm_module), comm_rank,
                         comm_size, comm->c_name, root));

    void *memory_map = sharm_module->local_op_memory_map;
    size_t *collectivies_info_bytes_received_by_rank = (size_t *) memory_map;

    ompi_datatype_type_size(datatype, &ddt_size);
    coll_info->sdtypes_contiguous[comm_rank][0]
        = ompi_datatype_is_contiguous_memory_layout(datatype, count);

    size_t total_size = ddt_size * count;

    int knem_fd = open(KNEM_DEVICE_FILENAME, O_RDWR);
    if (knem_fd == -1) {
        opal_output_verbose(
            SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
            "coll:sharm:%d:bcast_xpmem: (%d/%d/%s) can not open KNEM "
            "device, error code %d",
            SHARM_COLL(bcast, sharm_module), comm_rank, comm_size,
            comm->c_name, knem_fd);
        return OMPI_ERROR;
    }

    /*
     * Create region only if we have contigous datatype for this rank
     */
    if (root == comm_rank
        && coll_info->sdtypes_contiguous[comm_rank][0]) {
        /* Create KNEM shared memory region */
        struct knem_cmd_create_region knem_cmd;
        struct knem_cmd_param_iovec knem_iov[1];
        knem_iov[0].base = buf;
        knem_iov[0].len = bufsize;
        knem_cmd.iovec_array = (uintptr_t) &knem_iov[0];
        knem_cmd.iovec_nr = 1;
        knem_cmd.flags = 0;              /* KNEM_FLAG_SINGLEUSE; */
        knem_cmd.protection = PROT_READ; /* only allow remote readers */
        int rc = ioctl(knem_fd, KNEM_CMD_CREATE_REGION, &knem_cmd);
        if (rc != 0) {
            opal_output_verbose(
                SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                "coll:sharm:%d:bcast_xpmem: (%d/%d/%s) can not create "
                "shared memory region, error code %d",
                SHARM_COLL(bcast, sharm_module), comm_rank, comm_size,
                comm->c_name, rc);
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
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
        SHARM_PROFILING_TIME_START(sharm_module, bcast, push);
        int push = sharm_queue_push_contiguous(
            RESOLVE_COLLECTIVIES_DATA(sharm_module, comm_rank)
                + collectivies_info_bytes_sended,
            bytes_to_send, comm_rank, -1, comm, sharm_module);
        SHARM_PROFILING_TIME_STOP(sharm_module, bcast, push);
        collectivies_info_bytes_sended += push;
    }

    uint8_t is_contiguous_dtype = 1;
    for (int i = 0; i < comm_size; ++i) {
        is_contiguous_dtype &= coll_info->sdtypes_contiguous[i][0];
    }

    if (0 == is_contiguous_dtype) {
        opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                            "coll:sharm:%d:bcast_kmem: (%d/%d/%s) "
                            "Unsupported non-contigous datatype",
                            SHARM_COLL(bcast, sharm_module), comm_rank,
                            comm_size, comm->c_name);
        return OMPI_ERR_NOT_SUPPORTED;
    }

    if (root == comm_rank) {
        sharm_barrier_gather_cico(root, comm, module);

        err = ioctl(knem_fd, KNEM_CMD_DESTROY_REGION, &knem_cmd.cookie);
    } else {
        /* Wait for notification (for an cookie of the shared memory KNEM
         * region) */
        ptrdiff_t *knem_cookie = (ptrdiff_t *) data->shm_bcast_cma_addr[root];
        SHM_SPIN_CONDITION(*knem_cookie, bcast_knem_wait_addr_label);

        /* Read from a VM of the root */
        struct knem_cmd_inline_copy knem_icopy;
        struct knem_cmd_param_iovec knem_iov[1];
        knem_iov[0].base = buf;
        knem_iov[0].len = bufsize;
        knem_icopy.local_iovec_array = (uintptr_t) &knem_iov[0];
        knem_icopy.local_iovec_nr = 1;
        knem_icopy.remote_cookie = *knem_cookie;
        knem_icopy.remote_offset = 0;
        knem_icopy.write
            = 0; /* read from the remote region into our local segments */
        knem_icopy.flags = 0;
        int rc = ioctl(knem_fd, KNEM_CMD_INLINE_COPY, &knem_icopy);
        if (rc != 0) {
            opal_output_verbose(
                SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                "coll:sharm:%d:bcast_xpmem: (%d/%d/%s) can not copy from "
                "shared memory region, error code %d",
                SHARM_COLL(bcast, sharm_module), comm_rank, comm_size,
                comm->c_name, rc);
            return OMPI_ERROR;
        }

        sharm_barrier_gather_cico(root, comm, module);
    }

    OPAL_OUTPUT_VERBOSE(
        (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
         "coll:sharm:%d:bcast_kmem: (%d/%d/%s), root %d bcast complete",
         SHARM_COLL(bcast, sharm_module), comm_rank, comm_size,
         comm->c_name, root));

    return OMPI_SUCCESS;
#endif
}
