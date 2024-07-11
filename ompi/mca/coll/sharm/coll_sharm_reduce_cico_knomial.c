/** @file */

#include "coll_sharm.h"

/*
 * External variables.
 */

extern int mca_coll_sharm_stream;
extern int mca_coll_sharm_reduce_knomial_radix;

/*
 * Local functions.
 */

/**
 * @brief shared-memory based algorithm for Reduce using CICO approach.
 * Use k-nomial tree of processes.
 * Support non-commutative operations.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_reduce_cico_knomial(const void *sbuf, void *rbuf, int count,
                              struct ompi_datatype_t *dtype,
                              struct ompi_op_t *op, int root,
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
    int parent = 0;
    int knomial_radix = mca_coll_sharm_reduce_knomial_radix;

    knomial_radix = knomial_radix > 1 ? knomial_radix : 2;
    knomial_radix = knomial_radix <= node_comm_size ? knomial_radix
                                                    : node_comm_size - 1;
    const char *_sbuf = sbuf;
    char *_rbuf = rbuf;
    if (MPI_IN_PLACE == sbuf) {
        _sbuf = rbuf;
    }
    char *_reduce_tree_send_buffer = _sbuf;
    opal_convertor_t rconvertor;
    OBJ_CONSTRUCT(&rconvertor, opal_convertor_t);
    opal_convertor_t sconvertor;
    OBJ_CONSTRUCT(&sconvertor, opal_convertor_t);

    OPAL_OUTPUT_VERBOSE(
        (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
         "coll:sharm:%d:reduce_cico_knomial: (%d/%d/%s) root %d, radix %d",
         SHARM_COLL(reduce, sharm_module), node_comm_rank, node_comm_size,
         comm->c_name, root, knomial_radix));

    size_t ddt_size, segsize, segment_ddt_bytes, zero = 0;
    int64_t segment_ddt_count;

    ompi_datatype_type_size(dtype, &ddt_size);
    ompi_datatype_type_extent(dtype, &extent);
    is_contiguous_dtype = ompi_datatype_is_contiguous_memory_layout(dtype,
                                                                    count);

    /*
     * segment_ddt_count is how many data elements can be placed into fragment.
     */
    segment_ddt_count = shm_data->mu_queue_fragment_size / ddt_size;
    segment_ddt_bytes = segment_ddt_count * ddt_size;
    segsize = opal_datatype_span(&dtype->super, segment_ddt_count, &gap);

    size_t total_size_one_rank = count * ddt_size;

    int level = 0x1;
    int nchilds = 0;
    /*
     * Determinate parent
     */
    while (level < node_comm_size) {
        if (node_comm_rank % (knomial_radix * level)) {
            parent = node_comm_rank / (knomial_radix * level)
                     * (knomial_radix * level);
            break;
        }
        level *= knomial_radix;
    }

    /*
     * Correcting level because it contain k^(nchilds-1)
     */
    level /= knomial_radix;

    int ilevel = level;
    while (ilevel > 0) {
        for (int r = 1; r < knomial_radix; r++) {
            int child = node_comm_rank + ilevel * r;
            if (child < node_comm_size) {
                nchilds++;
            }
        }
        ilevel /= knomial_radix;
    }

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:reduce_cico_knomial: "
                         "(%d/%d/%s) root %d, parent %d, nchilds %d",
                         SHARM_COLL(reduce, sharm_module), node_comm_rank,
                         node_comm_size, comm->c_name, root, parent, nchilds));

    void *memory_map = sharm_module->local_op_memory_map;
    memset(memory_map, 0, 2 * segsize + nchilds * sizeof(int));
    int *childs = (int *) memory_map;
    char *recv_temp_buffer = (char *) ((int *) childs + nchilds);
    char *reduce_temp_segment = (char *) ((char *) recv_temp_buffer + segsize);

    // recv_temp_buffer = recv_temp_buffer - gap;
    // reduce_temp_segment = reduce_temp_segment - gap;

    ilevel = level;
    int childnum = 0;
    while (ilevel > 0) {
        for (int r = 1; r < knomial_radix; r++) {
            int child = node_comm_rank + ilevel * r;
            if (child < node_comm_size) {
                childs[childnum] = child;
                OPAL_OUTPUT_VERBOSE(
                    (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                     "coll:sharm:%d:reduce_cico_knomial: "
                     "(%d/%d/%s) root %d, child %d",
                     SHARM_COLL(reduce, sharm_module), node_comm_rank,
                     node_comm_size, comm->c_name, root, childs[childnum]));
                childnum++;
            }
        }
        ilevel /= knomial_radix;
    }

    _reduce_tree_send_buffer = nchilds ? reduce_temp_segment : _sbuf;
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
                    recv_temp_buffer, 0, &rconvertor))) {
            return ret;
        }

        /*
         * Construct convertors to send messages.
         * If we have 0 childs, just send sbuf.
         */
        if (OMPI_SUCCESS
            != (ret = opal_convertor_copy_and_prepare_for_send(
                    ompi_mpi_local_convertor, &(dtype->super), count,
                    _reduce_tree_send_buffer, 0, &sconvertor))) {
            return ret;
        }
    }

    opal_convertor_t rootconvertor;
    OBJ_CONSTRUCT(&rootconvertor, opal_convertor_t);

    /*
     * If datatype is not contigous - use pack/unpack methods.
     */
    if (0 == is_contiguous_dtype && root > 0 && node_comm_rank == root) {
        if (OMPI_SUCCESS
            != (ret = opal_convertor_copy_and_prepare_for_recv(
                    ompi_mpi_local_convertor, &(dtype->super), count, _rbuf, 0,
                    &rootconvertor))) {
            return ret;
        }
    }

    int64_t total_counts = count;
    char is_first = SHARM_TRUE;
    int fragment_num = 0;

    /*
     * Receive from childs, send to parent.
     */
    size_t bytes_received = 0;
    size_t bytes_sended = 0;

    /*
     * if root > 0 - root need to receive from 0.
     * Count it separately in same cycle.
     */
    size_t root_bytes_received = !(root > 0 && node_comm_rank == root);
    size_t root_rtotal_size = root_bytes_received ? 0 : total_size_one_rank;
    size_t root_bytes_sended = 0;
    while (bytes_received < total_size_one_rank
           || root_bytes_received < root_rtotal_size) {
        int64_t min_counts = min(total_counts, segment_ddt_count);
        /*
         * Recv-send from childs to parent.
         */
        if (bytes_received < total_size_one_rank) {
            is_first = SHARM_TRUE;
            for (int i = 0; i < nchilds; i++) {
                int pop = 0;
                if (is_contiguous_dtype) {
                    /*
                     * Hack - use buffer directly without copy.
                     */
                    wait_queue_func(pop, sharm_queue_get_ctrl(childs[i], comm,
                                                              sharm_module));
                    recv_temp_buffer = (char *)
                        sharm_queue_get_ptr(childs[i], comm, sharm_module);
                } else {
                    SHARM_PROFILING_TIME_START(sharm_module, reduce, pop);
                    wait_queue_func(pop,
                                    sharm_queue_pop(&(rconvertor), childs[i],
                                                    comm, sharm_module));
                    SHARM_PROFILING_TIME_STOP(sharm_module, reduce, pop);
                    opal_convertor_set_position(&(rconvertor), &zero);
                }

                if (SHARM_FALSE == is_first) {
                    SHARM_PROFILING_TIME_START(sharm_module, reduce,
                                               reduce_operation);
                    ompi_op_reduce(op, recv_temp_buffer, reduce_temp_segment,
                                   min_counts, dtype);
                    SHARM_PROFILING_TIME_STOP(sharm_module, reduce,
                                              reduce_operation);
                } else {
                    is_first = SHARM_FALSE;
                    SHARM_PROFILING_TIME_START(sharm_module, reduce, copy);
                    ompi_datatype_copy_content_same_ddt(dtype, min_counts,
                                                        reduce_temp_segment,
                                                        recv_temp_buffer);
                    SHARM_PROFILING_TIME_STOP(sharm_module, reduce, copy);
                }
                /*
                 * We complete data processing - clear control and release slot.
                 * Hack - use buffer directly without copy.
                 */
                if (is_contiguous_dtype) {
                    sharm_queue_clear_ctrl(childs[i], comm, sharm_module);
                }
            }

            /*
             * Reduce for our owned data if we received any fragmens from
             * childs.
             */
            if (nchilds) {
                SHARM_PROFILING_TIME_START(sharm_module, reduce,
                                           reduce_operation);
                ompi_op_reduce(op,
                               (char *) _sbuf
                                   + fragment_num * extent * segment_ddt_count,
                               reduce_temp_segment, min_counts, dtype);
                SHARM_PROFILING_TIME_STOP(sharm_module, reduce,
                                          reduce_operation);
            }

            /*
             * Send data to parent.
             */
            if (node_comm_rank > 0) {
                int push = 0;
                if (is_contiguous_dtype) {
                    int bytes_to_send = min(total_size_one_rank - bytes_sended,
                                            segment_ddt_bytes);
                    SHARM_PROFILING_TIME_START(sharm_module, reduce, push);
                    wait_queue_func(push, sharm_queue_push_contiguous(
                                              _reduce_tree_send_buffer
                                                  + (!nchilds) * bytes_sended,
                                              bytes_to_send, node_comm_rank,
                                              parent, comm, sharm_module));
                    SHARM_PROFILING_TIME_STOP(sharm_module, reduce, push);
                    bytes_sended += push;
                } else {
                    if (nchilds)
                        opal_convertor_set_position(&(sconvertor), &zero);

                    SHARM_PROFILING_TIME_START(sharm_module, reduce, push);
                    wait_queue_func(push,
                                    sharm_queue_push(&sconvertor,
                                                     segment_ddt_bytes,
                                                     node_comm_rank, parent,
                                                     comm, sharm_module));
                    SHARM_PROFILING_TIME_STOP(sharm_module, reduce, push);
                }
            }
            if (root == 0 && node_comm_rank == 0) {
                SHARM_PROFILING_TIME_START(sharm_module, reduce, copy);
                ompi_datatype_copy_content_same_ddt(
                    dtype, min_counts,
                    _rbuf + fragment_num * extent * segment_ddt_count,
                    reduce_temp_segment);
                SHARM_PROFILING_TIME_STOP(sharm_module, reduce, copy);
            }
            bytes_received += min_counts * ddt_size;
            total_counts -= segment_ddt_count;
            fragment_num++;
        }

        /*
         * If root is not rank 0
         */
        if (root > 0) {
            if (node_comm_rank == root) {
                int pop = 0;
                if (is_contiguous_dtype) {
                    SHARM_PROFILING_TIME_START(sharm_module, reduce, pop);
                    pop = sharm_queue_pop_contiguous(_rbuf
                                                         + root_bytes_received,
                                                     0, comm, sharm_module);
                    SHARM_PROFILING_TIME_STOP(sharm_module, reduce, pop);
                } else {
                    SHARM_PROFILING_TIME_START(sharm_module, reduce, pop);
                    pop = sharm_queue_pop(&(rootconvertor), 0, comm,
                                          sharm_module);
                    SHARM_PROFILING_TIME_STOP(sharm_module, reduce, pop);
                }
                root_bytes_received += pop;
            } else if (0 == node_comm_rank) {
                int push = 0;
                if (is_contiguous_dtype) {
                    int bytes_to_send = min(total_size_one_rank
                                                - root_bytes_sended,
                                            segment_ddt_bytes);

                    SHARM_PROFILING_TIME_START(sharm_module, reduce, push);
                    wait_queue_func(push,
                                    sharm_queue_push_contiguous(
                                        _reduce_tree_send_buffer
                                            + (!nchilds) * root_bytes_sended,
                                        bytes_to_send, node_comm_rank, root,
                                        comm, sharm_module));
                    SHARM_PROFILING_TIME_STOP(sharm_module, reduce, push);
                    root_bytes_sended += push;
                } else {
                    opal_convertor_set_position(&(sconvertor), &zero);
                    SHARM_PROFILING_TIME_START(sharm_module, reduce, push);
                    wait_queue_func(push, sharm_queue_push(&sconvertor,
                                                           segment_ddt_bytes,
                                                           node_comm_rank, root,
                                                           comm, sharm_module));
                    SHARM_PROFILING_TIME_STOP(sharm_module, reduce, push);
                }
            }
        }
    }

    int slots = ((total_size_one_rank + segment_ddt_bytes - 1)
                 / segment_ddt_bytes);

    // Adjust slots counters for sync it. Rank 0 did not send any data on
    // this moment
    for (int i = 0; i < node_comm_size; ++i) {
        if (i == node_comm_rank)
            continue;
        if (0 == i) {
            if (0 != root && 0 != node_comm_rank && node_comm_rank != root) {
                adjust_queue_current_slot(i, 0, slots, sharm_module);
            }
            continue;
        }
        char is_mychild = SHARM_FALSE;
        for (int j = 0; j < nchilds; j++) {
            if (childs[j] == i) {
                is_mychild = SHARM_TRUE;
                break;
            }
        }
        if (SHARM_FALSE == is_mychild)
            adjust_queue_current_slot(i, 0, slots, sharm_module);
    }

    opal_atomic_wmb();

    OBJ_DESTRUCT(&rconvertor);
    OBJ_DESTRUCT(&rootconvertor);
    OBJ_DESTRUCT(&sconvertor);

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:reduce_cico_knomial: (%d/%d/%s), root "
                         "%d reduce complete",
                         SHARM_COLL(reduce, sharm_module), node_comm_rank,
                         node_comm_size, comm->c_name, root));

    return OMPI_SUCCESS;
}
