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
 * @brief shared-memory based algorithm for Exscan using CICO approach.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_exscan_cico(const void *sbuf, void *rbuf, int count,
                      struct ompi_datatype_t *dtype, struct ompi_op_t *op,
                      struct ompi_communicator_t *comm,
                      mca_coll_base_module_t *module)
{
    int ret = 0;
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;
    ptrdiff_t extent;
    size_t ddt_size = 0;
    int64_t segment_ddt_count = 0;
    size_t segment_ddt_bytes = 0;
    size_t zero = 0;
    size_t total_size = 0;

    int node_comm_rank = ompi_comm_rank(comm);
    int node_comm_size = ompi_comm_size(comm);

    const char *_sbuf = sbuf;
    char *_rbuf = rbuf;

    if (MPI_IN_PLACE == sbuf) {
        _sbuf = rbuf;
    }

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:exscan_cico: (%d/%d/%s)",
                         SHARM_OP(sharm_module), node_comm_rank, node_comm_size,
                         comm->c_name));

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

    if (node_comm_rank > 0) {
        void *memory_map = sharm_module->local_op_memory_map;
        char *recv_temp_buffer = (char *) (memory_map);

        /*
         * Construct convertors to recv messages.
         */
        if (OMPI_SUCCESS
            != (ret = opal_convertor_copy_and_prepare_for_recv(
                    ompi_mpi_local_convertor, &(dtype->super), count, _rbuf, 0,
                    &rconvertor))) {
            return ret;
        }

        /*
         * Receive message.
         */
        size_t bytes_received = 0;
        while (bytes_received < total_size) {
            int pop = 0;
            wait_queue_func(pop,
                            sharm_queue_pop(&(rconvertor), node_comm_rank - 1,
                                            comm, sharm_module));
            bytes_received += pop;
        }

        /*
         * Construct convertors to send messages.
         */
        if (OMPI_SUCCESS
            != (ret = opal_convertor_copy_and_prepare_for_send(
                    ompi_mpi_local_convertor, &(dtype->super), count,
                    recv_temp_buffer, 0, &sconvertor))) {
            return ret;
        }

        /*
         * Send message.
         */
        size_t bytes_sended = 0;
        int64_t total_counts = count;
        while (bytes_sended < total_size) {
            int64_t min_counts = min(total_counts, segment_ddt_count);
            /*
             * Copy the send buffer into the receive buffer.
             */

            ret = ompi_datatype_copy_content_same_ddt(dtype, min_counts,
                                                      (char *) recv_temp_buffer,
                                                      (char *) _sbuf
                                                          + bytes_sended);
            /*
             * Reduce data with received result.
             */
            if (node_comm_rank < (node_comm_size - 1)) {
                ompi_op_reduce(op, _rbuf + bytes_sended, recv_temp_buffer,
                               min_counts, dtype);
            }
            opal_convertor_set_position(&(sconvertor), &zero);
            int push = sharm_queue_push(&sconvertor,
                                        shm_data->mu_queue_fragment_size,
                                        node_comm_rank, node_comm_rank + 1,
                                        comm, sharm_module);
            bytes_sended += push;
            total_counts -= min_counts;
        }
    }

    /*
     * Send result to next process.
     */
    if (node_comm_rank == 0) {
        /*
         * Construct convertors to send messages.
         */
        if (OMPI_SUCCESS
            != (ret = opal_convertor_copy_and_prepare_for_send(
                    ompi_mpi_local_convertor, &(dtype->super), count, _sbuf, 0,
                    &sconvertor))) {
            return ret;
        }

        /*
         * Send message.
         */
        size_t bytes_sended = 0;
        while (bytes_sended < total_size) {
            int push = sharm_queue_push(&sconvertor,
                                        shm_data->mu_queue_fragment_size,
                                        node_comm_rank, node_comm_rank + 1,
                                        comm, sharm_module);
            bytes_sended += push;
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

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:exscan_cico: "
                         "(%d/%d/%s), exscan complete",
                         SHARM_OP(sharm_module), node_comm_rank, node_comm_size,
                         comm->c_name));

    return OMPI_SUCCESS;
}
