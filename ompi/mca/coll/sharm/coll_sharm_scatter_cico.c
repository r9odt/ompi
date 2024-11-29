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
 * @brief shared-memory based algorithm for Scatter using CICO approach.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_scatter_cico(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                       void *rbuf, int rcount, ompi_datatype_t *rdtype,
                       int root, ompi_communicator_t *comm,
                       mca_coll_base_module_t *module)
{
    int ret = 0;
    size_t total_size = 0;
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;

    int comm_rank = ompi_comm_rank(comm);
    int comm_size = ompi_comm_size(comm);

    const char *_sbuf = sbuf;
    int _rcount = rcount;
    ompi_datatype_t *_rdtype = rdtype;
    char *_rbuf = rbuf;

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:scatter_cico: (%d/%d/%s) root %d",
                         SHARM_COLL(scatter, sharm_module), comm_rank,
                         comm_size, comm->c_name, root));

    if (root == comm_rank) {
        void *memory_map = sharm_module->local_op_memory_map;

        size_t *total_sizes_by_rank = (size_t *) memory_map;
        size_t *send_bytes_by_rank = (size_t *) ((size_t *) total_sizes_by_rank
                                                 + comm_size);
        opal_convertor_t *root_convertors_by_rank
            = (opal_convertor_t *) ((size_t *) send_bytes_by_rank
                                    + comm_size);

        ptrdiff_t sext;
        ompi_datatype_type_extent(sdtype, &sext);
        size_t sdtype_size;
        ompi_datatype_type_size(sdtype, &sdtype_size);

        for (int i = 0; i < comm_size; ++i) {
            send_bytes_by_rank[i] = 0;
            char *send_buff_ptr_for_rank = (char *) _sbuf + sext * scount * i;

            OBJ_CONSTRUCT(&root_convertors_by_rank[i], opal_convertor_t);

            if (i == comm_rank) {
                if (rbuf != MPI_IN_PLACE) {
                    ompi_datatype_sndrcv((char *) send_buff_ptr_for_rank,
                                         scount, sdtype, _rbuf, _rcount,
                                         _rdtype);
                }
                total_sizes_by_rank[i] = 0;
                continue;
            }

            if (OMPI_SUCCESS
                != (ret = opal_convertor_copy_and_prepare_for_send(
                        ompi_mpi_local_convertor, &(sdtype->super), scount,
                        send_buff_ptr_for_rank, 0,
                        &root_convertors_by_rank[i]))) {
                return ret;
            }
            total_sizes_by_rank[i] = sdtype_size * scount;
            total_size += total_sizes_by_rank[i];
        }

        if (rbuf == MPI_IN_PLACE) {
            _rcount = scount;
            _rbuf = (char *) sbuf + sext * scount * comm_rank;
            _rdtype = sdtype;
        }

        size_t bytes_sended = 0;
        while (bytes_sended < total_size) {
            for (int i = 0; i < comm_size; ++i) {
                if (send_bytes_by_rank[i] >= total_sizes_by_rank[i]) {
                    continue;
                }

                int push = 0;
                push = sharm_queue_push_to_subqueue(
                    &(root_convertors_by_rank[i]),
                    shm_data->mu_queue_fragment_size, root, i, i, comm,
                    sharm_module);
                bytes_sended += push;
                send_bytes_by_rank[i] += push;
            }
        }

        for (int i = 0; i < comm_size; ++i) {
            OBJ_DESTRUCT(&(root_convertors_by_rank[i]));
        }
    } else {
        opal_convertor_t convertor;
        OBJ_CONSTRUCT(&convertor, opal_convertor_t);
        if (OMPI_SUCCESS
            != (ret = opal_convertor_copy_and_prepare_for_recv(
                    ompi_mpi_local_convertor, &(_rdtype->super), _rcount, _rbuf,
                    0, &convertor))) {
            return ret;
        }

        size_t rdtype_size;
        ompi_datatype_type_size(rdtype, &rdtype_size);
        total_size = rdtype_size * rcount;

        int slots = ((total_size + shm_data->mu_queue_fragment_size - 1)
                     / shm_data->mu_queue_fragment_size);

        /*
         * Receive message.
         */
        size_t bytes_received = 0;
        while (bytes_received < total_size) {
            int pop = sharm_queue_pop_from_subqueue(&convertor, root,
                                                    comm_rank, comm,
                                                    sharm_module);
            bytes_received += pop;
        }

        // Adjust slots counters for sync it.
        for (int i = 0; i < comm_size; ++i) {
            if (i == comm_rank || i == root) {
                continue;
            }
            adjust_queue_current_slot(root, i, slots, sharm_module);
        }
        OBJ_DESTRUCT(&convertor);
    }

    opal_atomic_wmb();

    OPAL_OUTPUT_VERBOSE(
        (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
         "coll:sharm:%d:scatter_cico: (%d/%d/%s), root %d scatter complete",
         SHARM_COLL(scatter, sharm_module), comm_rank, comm_size,
         comm->c_name, root));

    return OMPI_SUCCESS;
}
