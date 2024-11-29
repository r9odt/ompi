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
 * @brief shared-memory based algorithm for Reduce using CICO approach.
 * Use flat tree of processes.
 * Support non-commutative operations.
 */
int sharm_reduce_cico_non_commutative(const void *sbuf, void *rbuf, int count,
                                      struct ompi_datatype_t *dtype,
                                      struct ompi_op_t *op, int root,
                                      struct ompi_communicator_t *comm,
                                      mca_coll_base_module_t *module)
{
    int ret = 0;
    size_t rtotal_size = 0;
    size_t stotal_size = 0;
    ptrdiff_t extent;
    uint8_t is_contiguous_dtype = 0;
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;

    int comm_rank = ompi_comm_rank(comm);
    int comm_size = ompi_comm_size(comm);

    const char *_sbuf = sbuf;
    char *_rbuf = rbuf;

    opal_convertor_t convertor;
    OBJ_CONSTRUCT(&convertor, opal_convertor_t);

    OPAL_OUTPUT_VERBOSE(
        (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
         "coll:sharm:%d:reduce_cico_non_commutative: (%d/%d/%s) root %d",
         SHARM_COLL(reduce, sharm_module), comm_rank, comm_size,
         comm->c_name, root));

    size_t ddt_size, segsize, segment_ddt_bytes, zero = 0;
    int64_t segment_ddt_count;

    ompi_datatype_type_size(dtype, &ddt_size);
    ompi_datatype_type_extent(dtype, &extent);

    /*
     * segment_ddt_count is how many data elements can be placed into fragment.
     */
    segment_ddt_count = shm_data->mu_queue_fragment_size / ddt_size;
    segment_ddt_bytes = segment_ddt_count * ddt_size;
    segsize = shm_data->mu_queue_fragment_size;
    // segsize = opal_datatype_span(&dtype->super, segment_ddt_count, &gap);
    is_contiguous_dtype = ompi_datatype_is_contiguous_memory_layout(dtype,
                                                                    count);

    if (root == comm_rank) {
        size_t total_size_one_rank = count * ddt_size;
        size_t *recv_bytes_by_rank = NULL;

        void *memory_map = sharm_module->local_op_memory_map;
        memset(memory_map, 0, 2 * segsize + comm_size * (sizeof(size_t)));

        recv_bytes_by_rank = (size_t *) memory_map;
        char *recv_temp_buffer = (char *) ((size_t *) recv_bytes_by_rank
                                           + comm_size);
        char *reduce_temp_segment = (char *) ((char *) recv_temp_buffer
                                              + segsize);

        /*
         * If datatype is not contigous - use pack/unpack methods.
         */
        if (0 == is_contiguous_dtype) {
            /*
             * Construct convertors to recv messages.
             */
            if (OMPI_SUCCESS
                != (ret = opal_convertor_copy_and_prepare_for_recv(
                        ompi_mpi_local_convertor, &(dtype->super), count,
                        recv_temp_buffer, 0, &convertor))) {
                return ret;
            }
        }

        int64_t total_counts = count;
        int fragment_num = 0;
        rtotal_size = total_size_one_rank * comm_size;
        for (int i = 0; i < comm_size; ++i) {
            recv_bytes_by_rank[i] = 0;
        }

        /*
         * If sbuf is MPI_IN_PLACE - send data from rbuf.
         */
        if (MPI_IN_PLACE == sbuf) {
            _sbuf = rbuf;
        }

        size_t bytes_received = 0;
        while (bytes_received < rtotal_size) {
            int64_t min_counts = min(total_counts, segment_ddt_count);
            for (int i = comm_size - 1; i >= 0; --i) {
                int pop = 0;
                /*
                 * If we receive any data - do partial reduction.
                 */
                if (OPAL_LIKELY(i != comm_rank)) {
                    if (is_contiguous_dtype) {
                        /*
                         * Hack - use buffer directly without copy.
                         */
                        wait_queue_func(pop,
                                        sharm_queue_get_ctrl(i, comm,
                                                             sharm_module));
                        recv_temp_buffer = (char *)
                            sharm_queue_get_ptr(i, comm, sharm_module);
                    } else {
                        wait_queue_func(pop,
                                        sharm_queue_pop(&(convertor), i, comm,
                                                        sharm_module));
                        opal_convertor_set_position(&(convertor), &zero);
                    }
                } else {
                    /*
                     * Hack - use buffer directly without copy.
                     */
                    if (is_contiguous_dtype) {
                        recv_temp_buffer = (char *) _sbuf
                                           + fragment_num * extent
                                                 * segment_ddt_count;
                    } else {
                        /*
                         * If we need to receive from self - just copy directly.
                         */
                        ompi_datatype_copy_content_same_ddt(
                            dtype, min_counts, recv_temp_buffer,
                            (char *) _sbuf
                                + fragment_num * extent * segment_ddt_count);
                    }
                    pop = min_counts * ddt_size;
                }
                bytes_received += pop;
                recv_bytes_by_rank[i] += pop;

                /*
                 * If we receive data from ranks lesser than last rank - partial
                 * reduce.
                 */
                if (OPAL_LIKELY(i < comm_size - 1)) {
                    ompi_op_reduce(op, recv_temp_buffer, reduce_temp_segment,
                                   min_counts, dtype);
                } else {
                    /*
                     * If we receive data from last rank - copy part from
                     * recv_temp
                     */
                    ompi_datatype_copy_content_same_ddt(dtype, min_counts,
                                                        reduce_temp_segment,
                                                        recv_temp_buffer);
                }
                /*
                 * We complete data processing - clear control and release slot.
                 * Hack - use buffer directly without copy.
                 */
                if ((i != comm_rank) && is_contiguous_dtype) {
                    sharm_queue_clear_ctrl(i, comm, sharm_module);
                }
            }

            /*
             * Copy result of partial reduce to rbuf.
             */
            ompi_datatype_copy_content_same_ddt(dtype, min_counts,
                                                _rbuf
                                                    + fragment_num * extent
                                                          * segment_ddt_count,
                                                reduce_temp_segment);
            total_counts -= segment_ddt_count;
            fragment_num++;
        }
    } else {
        stotal_size = count * ddt_size;
        if (0 == is_contiguous_dtype) {
            if (OMPI_SUCCESS
                != (ret = opal_convertor_copy_and_prepare_for_send(
                        ompi_mpi_local_convertor, &(dtype->super), count, _sbuf,
                        0, &convertor))) {
                return ret;
            }
        }

        /*
         * Send message.
         */
        size_t bytes_sended = 0;
        while (bytes_sended < stotal_size) {
            int push = 0;
            if (is_contiguous_dtype) {
                int bytes_to_send = min(stotal_size - bytes_sended,
                                        segment_ddt_bytes);

                push = sharm_queue_push_contiguous(_sbuf + bytes_sended,
                                                   bytes_to_send,
                                                   comm_rank, root, comm,
                                                   sharm_module);
            } else {
                push = sharm_queue_push(&convertor, segment_ddt_bytes,
                                        comm_rank, root, comm,
                                        sharm_module);
            }
            bytes_sended += push;
        }

        int slots = ((stotal_size + segment_ddt_bytes - 1) / segment_ddt_bytes);

        // Adjust slots counters for sync it.
        for (int i = 0; i < comm_size; ++i) {
            if (i == comm_rank || i == root) {
                continue;
            }
            adjust_queue_current_slot(i, 0, slots, sharm_module);
        }
    }

    opal_atomic_wmb();

    OBJ_DESTRUCT(&convertor);

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:reduce_cico_non_commutative: "
                         "(%d/%d/%s), root %d reduce complete",
                         SHARM_COLL(reduce, sharm_module), comm_rank,
                         comm_size, comm->c_name, root));
    return OMPI_SUCCESS;
}
