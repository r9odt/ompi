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
 * @brief shared-memory based algorithm for Scan using CMA approach.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_scan_cma(const void *sbuf, void *rbuf, int count,
                   struct ompi_datatype_t *dtype, struct ompi_op_t *op,
                   struct ompi_communicator_t *comm,
                   mca_coll_base_module_t *module)
{
    SHARM_INIT_PROFILING_COUNTERS();
    int ret = 0;
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;
    ptrdiff_t extent;
    size_t ddt_size, segment_ddt_bytes, zero = 0;
    int64_t segment_ddt_count;
    size_t total_size = 0;

    int comm_rank = ompi_comm_rank(comm);
    int comm_size = ompi_comm_size(comm);

    const char *_sbuf = sbuf;
    char *_rbuf = rbuf;

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:scan_cma: (%d/%d/%s)",
                         SHARM_COLL(scan, sharm_module), comm_rank, comm_size,
                         comm->c_name));

    if (MPI_IN_PLACE == sbuf) {
        _sbuf = rbuf;
    }

    /*
     * Use collectivies exchange v2.1
     */

    char *my_coll_info_block = RESOLVE_COLLECTIVIES_DATA(sharm_module,
                                                         comm_rank);
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
           comm_size * sizeof(size_t));
    size_t collectivies_info_bytes_sended = 0;
    size_t collectivies_info_bytes_received = 0;
    size_t collectivies_info_bytes_total_one_rank = sizeof(char)
                                                    + sizeof(ptrdiff_t);
    size_t collectivies_info_bytes_total
        = (comm_size - 1) * collectivies_info_bytes_total_one_rank;
    while (
        collectivies_info_bytes_sended < collectivies_info_bytes_total_one_rank
        || collectivies_info_bytes_received < collectivies_info_bytes_total) {
        for (int i = 0; i < comm_size; ++i) {
            if (i == comm_rank
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
            comm_rank, -1, comm, sharm_module);
        SHARM_PROFILING_TIME_STOP(sharm_module, scan, push);
        collectivies_info_bytes_sended += push;
    }

    uint8_t is_contiguous_dtype = 1;
    for (int i = 0; i < comm_size; ++i) {
        char *coll_info_is_contigous = RESOLVE_COLLECTIVIES_DATA(sharm_module,
                                                                 i);
        is_contiguous_dtype &= *coll_info_is_contigous;
    }

    SHARM_PROFILING_TIME_STOP(sharm_module, scan, collective_exchange);

    if (0 == is_contiguous_dtype) {
        opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                            "coll:sharm:%d:reduce_cma: (%d/%d/%s) "
                            "Unsupported non-contigous datatype",
                            SHARM_COLL(reduce, sharm_module), comm_rank,
                            comm_size, comm->c_name);
        return OMPI_ERR_NOT_SUPPORTED;
    }

    uint8_t notify_data = 1;
    int push = 0;
    int pop = 0;
    ompi_datatype_type_size(dtype, &ddt_size);
    ompi_datatype_type_extent(dtype, &extent);
    total_size = ddt_size * count;

    /*
     * segment_ddt_count is how many data elements can be placed into fragment.
     */
    segment_ddt_count = shm_data->mu_queue_fragment_size / ddt_size;
    segment_ddt_bytes = segment_ddt_count * ddt_size;

    int recvfrom = comm_rank - 1;
    int sendto = comm_rank + 1;

    /*
     * If I'm rank 0, just copy into the receive buffer
     */
    if (0 == comm_rank) {
        if (MPI_IN_PLACE != sbuf) {
            ret = ompi_datatype_copy_content_same_ddt(dtype, count,
                                                      (char *) _rbuf,
                                                      (char *) _sbuf);
            if (MPI_SUCCESS != ret) {
                return ret;
            }
        }
    } else {
        /*
         * Otherwise receive previous buffer and reduce.
         */
        char *recv_temp_buffer = (char *) (memory_map);
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
         * Receive message.
         */
        int64_t total_counts = count;
        size_t bytes_received = 0;
        int fragment_num = 0;

        ptrdiff_t *peer_buff
            = (ptrdiff_t *) (RESOLVE_COLLECTIVIES_DATA(sharm_module, recvfrom)
                             + sizeof(char));

        while (total_counts > 0) {
            int64_t min_counts = min(total_counts, segment_ddt_count);

            int bytes_to_copy = min_counts * extent;
            int rc = sharm_cma_readv(SHARM_GET_RANK_PID(shm_data, recvfrom),
                                     recv_temp_buffer,
                                     (void *) (((char *) (*peer_buff))
                                               + fragment_num * extent
                                                     * segment_ddt_count),
                                     bytes_to_copy);

            bytes_received += pop;
            ompi_op_reduce(op, recv_temp_buffer,
                           _rbuf + fragment_num * extent * segment_ddt_count,
                           min_counts, dtype);
            fragment_num++;
            total_counts -= segment_ddt_count;
        }
    }

    /*
     * Notify next
     */
    if (comm_rank < (comm_size - 1)) {
        wait_queue_func(push, sharm_queue_push_contiguous(&notify_data, 1,
                                                          comm_rank, sendto, comm,
                                                          sharm_module));
    }
    
    for (int i = 0; i < comm_size - 1; ++i) {
        if (i == comm_rank || i == comm_rank - 1)
            continue;
        adjust_queue_current_slot(i, 0, 1, sharm_module);
    }

    opal_atomic_wmb();

    SHARM_PROFILING_TIME_START(sharm_module, scan, zcopy_barrier);
    sharm_barrier_intra(comm, module);
    SHARM_PROFILING_TIME_STOP(sharm_module, scan, zcopy_barrier);

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:scan_cma: "
                         "(%d/%d/%s), scan complete",
                         SHARM_COLL(scan, sharm_module), comm_rank, comm_size,
                         comm->c_name));
    return OMPI_SUCCESS;
}
