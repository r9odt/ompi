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
 * @brief shared-memory based algorithm for Reduce using XPMEM approach.
 * Use flat tree of processes.
 * Support non-commutative operations.
 */
int sharm_reduce_xpmem(const void *sbuf, void *rbuf, int count,
                       struct ompi_datatype_t *dtype, struct ompi_op_t *op,
                       int root, struct ompi_communicator_t *comm,
                       mca_coll_base_module_t *module)
{
#if !(SHARM_CHECK_XPMEM_SUPPORT)
    return OMPI_ERR_NOT_AVAILABLE;
#else
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;

    int node_comm_rank = ompi_comm_rank(comm);
    int node_comm_size = ompi_comm_size(comm);

    const char *_sbuf = sbuf;
    char *_rbuf = rbuf;

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:reduce_xpmem: (%d/%d/%s) root %d",
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
                            "coll:sharm:%d:reduce_xpmem: (%d/%d/%s) "
                            "Unsupported non-contigous datatype",
                            SHARM_COLL(reduce, sharm_module), node_comm_rank,
                            node_comm_size, comm->c_name);
        return OMPI_ERR_NOT_SUPPORTED;
    }

    /*
     * Data exchange.
     */

    int64_t total_size = extent * count;
    if (root == node_comm_rank) {
        if (sbuf != MPI_IN_PLACE) {
            for (int i = node_comm_size - 1; i >= 0; --i) {
                if (OPAL_UNLIKELY(node_comm_rank == i)) {
                    ompi_op_reduce(op, _sbuf, _rbuf, count, dtype);
                    continue;
                }
                xpmem_segid_t segid = *(
                    (xpmem_segid_t *) shm_data->xpmem_segid[i]);
                xpmem_apid_t apid = -1;

                ptrdiff_t *peer_buff
                    = (ptrdiff_t *) (RESOLVE_COLLECTIVIES_DATA(sharm_module, i)
                                     + sizeof(char));
                // ptrdiff_t *peer_ext = ((char *) peer_buff) +
                // sizeof(ptrdiff_t);
                char *aligned_buff = (char *) (((uintptr_t) *peer_buff)
                                               & ~(shm_data->mu_page_size - 1));
                size_t align_offset = *peer_buff - (ptrdiff_t) aligned_buff;
                size_t aligned_total_size = (total_size + align_offset
                                             + shm_data->mu_page_size - 1)
                                            & ~(shm_data->mu_page_size - 1);
                SHARM_PROFILING_TIME_START(sharm_module, reduce, xpmem_attach);
                apid = xpmem_get(segid, XPMEM_RDWR, XPMEM_PERMIT_MODE, NULL);
                if (OPAL_UNLIKELY(apid < 0)) {
                    opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                                        "coll:sharm:%d:reduce_xpmem: "
                                        "(%d/%d/%s) can not get apid of"
                                        "shared memory region, error code %ld",
                                        SHARM_COLL(reduce, sharm_module),
                                        node_comm_rank, node_comm_size,
                                        comm->c_name, apid);
                    return OMPI_ERROR;
                }

                struct xpmem_addr addr = {.apid = apid,
                                          .offset = (off_t) aligned_buff};
                int64_t xpmem_seg_addr = (int64_t)
                    xpmem_attach(addr, aligned_total_size, (void *) NULL);

                if (OPAL_UNLIKELY(xpmem_seg_addr < -1)) {
                    opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                                        "coll:sharm:%d:reduce_xpmem: "
                                        "(%d/%d/%s) can not attach of"
                                        "shared memory region, error code %ld",
                                        SHARM_COLL(reduce, sharm_module),
                                        node_comm_rank, node_comm_size,
                                        comm->c_name, xpmem_seg_addr);
                    xpmem_release(apid);
                    return OMPI_ERROR;
                }
                SHARM_PROFILING_TIME_STOP(sharm_module, reduce, xpmem_attach);

                SHARM_PROFILING_TIME_START(sharm_module, reduce,
                                           reduce_operation);
                ompi_op_reduce(op, (void *) (xpmem_seg_addr + align_offset),
                               _rbuf, count, dtype);
                SHARM_PROFILING_TIME_STOP(sharm_module, reduce,
                                          reduce_operation);

                opal_atomic_wmb();

                xpmem_detach((void *) xpmem_seg_addr);
                xpmem_release(apid);
            }
        } else {
            char *reduce_temp_buffer = (char *) memory_map;
            int fragment_num = 0;
            int64_t total_counts = count;
            int64_t segment_ddt_count = shm_data->mu_queue_fragment_size
                                        / extent;
            while (total_counts > 0) {
                int64_t min_counts = min(total_counts, segment_ddt_count);
                for (int i = node_comm_size - 1; i >= 0; --i) {
                    if (OPAL_UNLIKELY(node_comm_rank == i)) {
                        if (i == node_comm_size - 1) {
                            SHARM_PROFILING_TIME_START(sharm_module, reduce,
                                                       copy);
                            ompi_datatype_copy_content_same_ddt(
                                dtype, min_counts, reduce_temp_buffer,
                                _sbuf
                                    + fragment_num * extent
                                          * segment_ddt_count);
                            SHARM_PROFILING_TIME_STOP(sharm_module, reduce,
                                                      copy);
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

                    xpmem_segid_t segid = *(
                        (xpmem_segid_t *) shm_data->xpmem_segid[i]);
                    xpmem_apid_t apid = -1;

                    ptrdiff_t *peer_buff
                        = (ptrdiff_t *) (RESOLVE_COLLECTIVIES_DATA(sharm_module,
                                                                   i)
                                         + sizeof(char));
                    // ptrdiff_t *peer_ext = ((char *) peer_buff)
                    //                       + sizeof(ptrdiff_t);
                    char *aligned_buff = (char *) (((uintptr_t) *peer_buff)
                                                   & ~(shm_data->mu_page_size
                                                       - 1));
                    size_t align_offset = *peer_buff - (ptrdiff_t) aligned_buff;
                    size_t aligned_total_size = (total_size + align_offset
                                                 + shm_data->mu_page_size - 1)
                                                & ~(shm_data->mu_page_size - 1);
                    SHARM_PROFILING_TIME_START(sharm_module, reduce,
                                               xpmem_attach);
                    apid = xpmem_get(segid, XPMEM_RDWR, XPMEM_PERMIT_MODE,
                                     NULL);
                    if (apid < 0) {
                        opal_output_verbose(
                            SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                            "coll:sharm:%d:reduce_xpmem: "
                            "(%d/%d/%s) can not get apid of"
                            "shared memory region, error code %ld",
                            SHARM_COLL(reduce, sharm_module), node_comm_rank,
                            node_comm_size, comm->c_name, apid);
                        return OMPI_ERROR;
                    }

                    struct xpmem_addr addr = {.apid = apid,
                                              .offset = (off_t) aligned_buff};
                    int64_t xpmem_seg_addr = (int64_t)
                        xpmem_attach(addr, aligned_total_size, (void *) NULL);

                    if (xpmem_seg_addr < -1) {
                        opal_output_verbose(
                            SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                            "coll:sharm:%d:reduce_xpmem: "
                            "(%d/%d/%s) can not attach of"
                            "shared memory region, error code %ld",
                            SHARM_COLL(reduce, sharm_module), node_comm_rank,
                            node_comm_size, comm->c_name, xpmem_seg_addr);
                        xpmem_release(apid);
                        return OMPI_ERROR;
                    }
                    SHARM_PROFILING_TIME_STOP(sharm_module, reduce,
                                              xpmem_attach);

                    if (i == node_comm_size - 1) {
                        SHARM_PROFILING_TIME_START(sharm_module, reduce, copy);
                        ompi_datatype_copy_content_same_ddt(
                            dtype, min_counts, reduce_temp_buffer,
                            (void *) (xpmem_seg_addr + align_offset
                                      + fragment_num * extent
                                            * segment_ddt_count));
                        SHARM_PROFILING_TIME_STOP(sharm_module, reduce, copy);
                    } else {
                        SHARM_PROFILING_TIME_START(sharm_module, reduce,
                                                   reduce_operation);
                        ompi_op_reduce(op,
                                       (void *) (xpmem_seg_addr + align_offset
                                                 + fragment_num * extent
                                                       * segment_ddt_count),
                                       reduce_temp_buffer, min_counts, dtype);
                        SHARM_PROFILING_TIME_STOP(sharm_module, reduce,
                                                  reduce_operation);
                    }

                    opal_atomic_wmb();

                    xpmem_detach((void *) xpmem_seg_addr);
                    xpmem_release(apid);
                }

                SHARM_PROFILING_TIME_START(sharm_module, reduce,
                                           reduce_operation);
                ompi_datatype_copy_content_same_ddt(
                    dtype, min_counts,
                    _rbuf + fragment_num * extent * segment_ddt_count,
                    reduce_temp_buffer);
                SHARM_PROFILING_TIME_STOP(sharm_module, reduce,
                                          reduce_operation);

                total_counts -= segment_ddt_count;
                fragment_num++;
            }
        }
    }

    SHARM_PROFILING_TIME_START(sharm_module, reduce, zcopy_barrier);
    sharm_barrier_sense_reversing(comm, module);
    SHARM_PROFILING_TIME_STOP(sharm_module, reduce, zcopy_barrier);

    OPAL_OUTPUT_VERBOSE(
        (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
         "coll:sharm:%d:reduce_xpmem: (%d/%d/%s), root %d reduce complete",
         SHARM_COLL(reduce, sharm_module), node_comm_rank, node_comm_size,
         comm->c_name, root));

    return OMPI_SUCCESS;
#endif
}
