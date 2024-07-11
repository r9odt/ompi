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
 * @brief shared-memory based algorithm for Allreduce using CICO approach.
 * Support non-commutative operations.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_allreduce_cico_non_commutative(const void *sbuf, void *rbuf,
                                         int count,
                                         struct ompi_datatype_t *dtype,
                                         struct ompi_op_t *op,
                                         struct ompi_communicator_t *comm,
                                         mca_coll_base_module_t *module)
{
    int ret = 0;
    ptrdiff_t extent, gap = 0;
    uint8_t is_contiguous_dtype = 0;
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;

    int node_comm_rank = ompi_comm_rank(comm);
    int node_comm_size = ompi_comm_size(comm);

    const char *_sbuf = sbuf;
    char *_rbuf = rbuf;

    opal_convertor_t sconvertor;
    OBJ_CONSTRUCT(&sconvertor, opal_convertor_t);
    opal_convertor_t rconvertor;
    OBJ_CONSTRUCT(&rconvertor, opal_convertor_t);

    OPAL_OUTPUT_VERBOSE(
        (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
         "coll:sharm:%d:allreduce_cico_non_commutative: (%d/%d/%s)",
         SHARM_OP(sharm_module), node_comm_rank, node_comm_size, comm->c_name));

    size_t ddt_size, segsize, segment_ddt_bytes, zero = 0;
    int64_t segment_ddt_count;

    ompi_datatype_type_size(dtype, &ddt_size);
    ompi_datatype_type_extent(dtype, &extent);

    /*
     * segment_ddt_count is how many data elements can be placed into fragment.
     */
    segment_ddt_count = shm_data->mu_queue_fragment_size / ddt_size;
    segment_ddt_bytes = segment_ddt_count * ddt_size;
    segsize = opal_datatype_span(&dtype->super, segment_ddt_count, &gap);
    is_contiguous_dtype = ompi_datatype_is_contiguous_memory_layout(dtype,
                                                                    count);
    size_t *recv_bytes_by_rank = NULL;

    void *memory_map = sharm_module->local_op_memory_map;
    memset(memory_map, 0, 2 * segsize + node_comm_size * (sizeof(size_t)));
    recv_bytes_by_rank = (size_t *) memory_map;
    char *recv_temp_buffer = (char *) ((size_t *) recv_bytes_by_rank
                                       + node_comm_size);
    char *reduce_temp_segment = (char *) ((char *) recv_temp_buffer + segsize);

    recv_temp_buffer = recv_temp_buffer - gap;
    reduce_temp_segment = reduce_temp_segment - gap;

    size_t rtotal_size = count * ddt_size;
    size_t stotal_size = rtotal_size;
    /*
     * If datatype is not contigous - use pack/unpack methods.
     */
    if (0 == is_contiguous_dtype) {
        /*
         * Construct convertors to send messages.
         */
        if (OMPI_SUCCESS
            != (ret = opal_convertor_copy_and_prepare_for_recv(
                    ompi_mpi_local_convertor, &(dtype->super), count,
                    recv_temp_buffer, 0, &rconvertor))) {
            return ret;
        }
        if (OMPI_SUCCESS
            != (ret = opal_convertor_copy_and_prepare_for_send(
                    ompi_mpi_local_convertor, &(dtype->super), count, _sbuf, 0,
                    &sconvertor))) {
            return ret;
        }
    }

    int64_t total_counts = count;
    int fragment_num = 0;
    for (int i = 0; i < node_comm_size; ++i) {
        recv_bytes_by_rank[i] = 0;
    }

    if (MPI_IN_PLACE == sbuf) {
        _sbuf = rbuf;
    }

    size_t bytes_received = 0;
    size_t bytes_sended = 0;
    rtotal_size = rtotal_size * node_comm_size;

    while (bytes_received < rtotal_size || bytes_sended < stotal_size) {
        if (bytes_sended < stotal_size) {
            int push = 0;
            if (is_contiguous_dtype) {
                int bytes_to_send = min(stotal_size - bytes_sended,
                                        shm_data->mu_queue_fragment_size);
                wait_queue_func(push,
                                sharm_queue_push_contiguous(_sbuf
                                                                + bytes_sended,
                                                            bytes_to_send,
                                                            node_comm_rank, -1,
                                                            comm,
                                                            sharm_module));
            } else {
                wait_queue_func(push, sharm_queue_push(&(sconvertor),
                                                       segment_ddt_bytes,
                                                       node_comm_rank, -1, comm,
                                                       sharm_module));
            }
            bytes_sended += push;
        }

        int64_t min_counts = min(total_counts, segment_ddt_count);
        for (int i = node_comm_size - 1; i >= 0; --i) {
            int pop = 0;
            /*
             * Receive data and do partial reduction.
             */
            if (OPAL_LIKELY((i != node_comm_rank))) {
                if (is_contiguous_dtype) {
                    wait_queue_func(pop,
                                    sharm_queue_pop_contiguous(recv_temp_buffer,
                                                               i, comm,
                                                               sharm_module));
                } else {
                    wait_queue_func(pop, sharm_queue_pop(&(rconvertor), i, comm,
                                                         sharm_module));
                    opal_convertor_set_position(&(rconvertor), &zero);
                }
            } else {
                ompi_datatype_copy_content_same_ddt(
                    dtype, min_counts, recv_temp_buffer,
                    (char *) _sbuf + fragment_num * extent * segment_ddt_count);
                pop = min_counts * ddt_size;
            }
            bytes_received += pop;
            recv_bytes_by_rank[i] += pop;
            /*
             * If we receive data from ranks lesser than last rank - partial
             * reduce.
             */
            if (OPAL_LIKELY(i < node_comm_size - 1)) {
                ompi_op_reduce(op, recv_temp_buffer, reduce_temp_segment,
                               min_counts, dtype);
            } else {
                /*
                 * If we receive data from last rank - copy part from.
                 * recv_temp
                 */
                ompi_datatype_copy_content_same_ddt(dtype, min_counts,
                                                    reduce_temp_segment,
                                                    recv_temp_buffer);
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

    OBJ_DESTRUCT(&sconvertor);
    OBJ_DESTRUCT(&rconvertor);

    opal_atomic_wmb();

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:allreduce_cico_non_commutative: "
                         "(%d/%d/%s), allreduce complete",
                         SHARM_OP(sharm_module), node_comm_rank, node_comm_size,
                         comm->c_name));
    return OMPI_SUCCESS;
}
