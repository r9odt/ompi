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
 * @brief shared-memory based algorithm for Scan using XPMEM approach.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_scan_xpmem(const void *sbuf, void *rbuf, int count,
                     struct ompi_datatype_t *dtype, struct ompi_op_t *op,
                     struct ompi_communicator_t *comm,
                     mca_coll_base_module_t *module)
{
#if !(SHARM_CHECK_XPMEM_SUPPORT)
    return OMPI_ERR_NOT_AVAILABLE;
#else
    int ret = 0;
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;
    ptrdiff_t extent;
    size_t ddt_size, segment_ddt_bytes, zero = 0;
    int64_t segment_ddt_count;
    size_t total_size = 0;

    int node_comm_rank = ompi_comm_rank(comm);
    int node_comm_size = ompi_comm_size(comm);

    const char *_sbuf = sbuf;
    char *_rbuf = rbuf;

    if (MPI_IN_PLACE == sbuf) {
        _sbuf = rbuf;
    }

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:scan_xpmem: (%d/%d/%s)",
                         SHARM_COLL(scan, sharm_module), node_comm_rank,
                         node_comm_size, comm->c_name));

    ompi_datatype_type_size(dtype, &ddt_size);
    ompi_datatype_type_extent(dtype, &extent);
    total_size = ddt_size * count;

    /*
     * segment_ddt_count is how many data elements can be placed into fragment.
     */
    segment_ddt_count = shm_data->mu_queue_fragment_size / ddt_size;
    segment_ddt_bytes = segment_ddt_count * ddt_size;
    opal_convertor_t sconvertor;
    OBJ_CONSTRUCT(&sconvertor, opal_convertor_t);
    opal_convertor_t rconvertor;
    OBJ_CONSTRUCT(&rconvertor, opal_convertor_t);

    char *my_coll_info_block = RESOLVE_COLLECTIVIES_DATA(sharm_module,
                                                         node_comm_rank);
    char *coll_info = my_coll_info_block;
    *my_coll_info_block = ompi_datatype_is_contiguous_memory_layout(dtype,
                                                                    count);
    coll_info = coll_info + sizeof(char);
    *(ptrdiff_t *) (coll_info) = (ptrdiff_t) _rbuf;

    void *memory_map = sharm_module->local_op_memory_map;
    size_t *collectivies_info_bytes_received_by_rank = (size_t *) memory_map;

    /*
     * Exchange collectivies info.
     */

    SHARM_PROFILING_TIME_START(sharm_module, scan, collective_exchange);
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
            SHARM_PROFILING_TIME_START(sharm_module, scan, pop);
            int pop = sharm_queue_pop_contiguous(
                RESOLVE_COLLECTIVIES_DATA(sharm_module, i)
                    + collectivies_info_bytes_received_by_rank[i],
                i, comm, sharm_module);
            SHARM_PROFILING_TIME_STOP(sharm_module, scan, pop);
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
        SHARM_PROFILING_TIME_START(sharm_module, scan, push);
        int push = sharm_queue_push_contiguous(
            my_coll_info_block + collectivies_info_bytes_sended, bytes_to_send,
            node_comm_rank, -1, comm, sharm_module);
        SHARM_PROFILING_TIME_STOP(sharm_module, scan, push);
        collectivies_info_bytes_sended += push;
    }

    uint8_t is_contiguous_dtype = 1;
    for (int i = 0; i < node_comm_size; ++i) {
        char *coll_info_is_contigous = RESOLVE_COLLECTIVIES_DATA(sharm_module,
                                                                 i);
        is_contiguous_dtype &= *coll_info_is_contigous;
    }

    SHARM_PROFILING_TIME_STOP(sharm_module, scan, collective_exchange);

    if (0 == is_contiguous_dtype) {
        opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                            "coll:sharm:%d:scan_xpmem: (%d/%d/%s) "
                            "Unsupported non-contigous datatype",
                            SHARM_COLL(scan, sharm_module), node_comm_rank,
                            node_comm_size, comm->c_name);
        return OMPI_ERR_NOT_SUPPORTED;
    }

    uint8_t notify_data = 1;
    int push = 0;
    int pop = 0;

    /*
     * If I'm rank 0, just copy into the receive buffer and notify next
     */
    if (0 == node_comm_rank) {
        if (MPI_IN_PLACE != sbuf) {
            ret = ompi_datatype_copy_content_same_ddt(dtype, count,
                                                      (char *) _rbuf,
                                                      (char *) _sbuf);
            if (MPI_SUCCESS != ret) {
                return ret;
            }
        }
        wait_queue_func(push, sharm_queue_push_contiguous(&notify_data, 1,
                                                          node_comm_rank + 1,
                                                          node_comm_rank + 1,
                                                          comm, sharm_module))
    } else {
        int recvfrom = node_comm_rank - 1;
        int sendto = node_comm_rank + 1;
        wait_queue_func(pop,
                        sharm_queue_get_ctrl(recvfrom, comm, sharm_module));
        sharm_queue_clear_ctrl(recvfrom, comm, sharm_module);

        /*
         * Copy the send buffer into the receive buffer.
         */
        if (MPI_IN_PLACE != _sbuf) {
            ret = ompi_datatype_copy_content_same_ddt(dtype, count,
                                                      (char *) _rbuf,
                                                      (char *) _sbuf);
            if (MPI_SUCCESS != ret) {
                return ret;
            }
        }

        /*
         * Map memory.
         */
        xpmem_segid_t segid = *(
            (xpmem_segid_t *) shm_data->xpmem_segid[recvfrom]);
        xpmem_apid_t apid = -1;

        ptrdiff_t *peer_buff
            = (ptrdiff_t *) (RESOLVE_COLLECTIVIES_DATA(sharm_module, recvfrom)
                             + sizeof(char));
        char *aligned_buff = (char *) (((uintptr_t) *peer_buff)
                                       & ~(shm_data->mu_page_size - 1));
        size_t align_offset = *peer_buff - (ptrdiff_t) aligned_buff;
        size_t aligned_total_size = (total_size + align_offset
                                     + shm_data->mu_page_size - 1)
                                    & ~(shm_data->mu_page_size - 1);
        SHARM_PROFILING_TIME_START(sharm_module, scan, xpmem_attach);
        apid = xpmem_get(segid, XPMEM_RDWR, XPMEM_PERMIT_MODE, NULL);
        if (OPAL_UNLIKELY(apid < 0)) {
            opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                                "coll:sharm:%d:scan_xpmem: "
                                "(%d/%d/%s) can not get apid of"
                                "shared memory region, error code %ld",
                                SHARM_COLL(scan, sharm_module), node_comm_rank,
                                node_comm_size, comm->c_name, apid);
            return OMPI_ERROR;
        }

        struct xpmem_addr addr = {.apid = apid, .offset = (off_t) aligned_buff};
        int64_t xpmem_seg_addr = (int64_t) xpmem_attach(addr,
                                                        aligned_total_size,
                                                        (void *) NULL);

        if (OPAL_UNLIKELY(xpmem_seg_addr < -1)) {
            opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                                "coll:sharm:%d:scan_xpmem: "
                                "(%d/%d/%s) can not attach of"
                                "shared memory region, error code %ld",
                                SHARM_COLL(scan, sharm_module), node_comm_rank,
                                node_comm_size, comm->c_name, xpmem_seg_addr);
            xpmem_release(apid);
            return OMPI_ERROR;
        }
        SHARM_PROFILING_TIME_STOP(sharm_module, scan, xpmem_attach);

        /*
         * Do reduce.
         */
        ompi_op_reduce(op, (void *) (xpmem_seg_addr + align_offset), _rbuf,
                       count, dtype);
        /*
         * Notify next
         */
        if (node_comm_rank < (node_comm_size - 1)) {
            wait_queue_func(push,
                            sharm_queue_push_contiguous(&notify_data, 1,
                                                        node_comm_rank + 1,
                                                        node_comm_rank + 1,
                                                        comm, sharm_module))
        }
    }

    for (int i = 0; i < node_comm_size - 1; ++i) {
        if (i == node_comm_rank - 1 || i == node_comm_rank)
            continue;
        adjust_queue_current_slot(i, 0,
                                  (total_size + segment_ddt_bytes - 1)
                                      / segment_ddt_bytes,
                                  sharm_module);
    }

    OBJ_DESTRUCT(&sconvertor);
    OBJ_DESTRUCT(&rconvertor);

    opal_atomic_wmb();

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:scan_xpmem: "
                         "(%d/%d/%s), scan complete",
                         SHARM_COLL(scan, sharm_module), node_comm_rank,
                         node_comm_size, comm->c_name));
    return OMPI_SUCCESS;
#endif
}
