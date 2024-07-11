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
 * @brief shared-memory based algorithm for Gatherv using CICO approach.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_gatherv_cico(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                       void *rbuf, const int *rcounts, const int *displs,
                       ompi_datatype_t *rdtype, int root,
                       ompi_communicator_t *comm,
                       mca_coll_base_module_t *module)
{
    int ret = 0;
    size_t total_size = 0;
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;
    sharm_local_collectivies_data_t *coll_info = &(
        sharm_module->local_collectivies_info);

    int node_comm_rank = ompi_comm_rank(comm);
    int node_comm_size = ompi_comm_size(comm);

    const char *_sbuf = sbuf;
    int _scount = scount;
    ompi_datatype_t *_sdtype = sdtype;
    char *_rbuf = rbuf;

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:gatherv_cico: (%d/%d/%s) root %d",
                         SHARM_COLL(gatherv, sharm_module), node_comm_rank,
                         node_comm_size, comm->c_name, root));

    if (root == node_comm_rank) {
        void *memory_map = sharm_module->local_op_memory_map;
        size_t *total_sizes_by_rank = (size_t *) memory_map;
        size_t *recv_bytes_by_rank = (size_t *) ((size_t *) total_sizes_by_rank
                                                 + node_comm_size);
        opal_convertor_t *root_convertors_by_rank
            = (opal_convertor_t *) ((size_t *) recv_bytes_by_rank
                                    + node_comm_size);

        size_t rdtype_size;
        ompi_datatype_type_size(rdtype, &rdtype_size);
        ptrdiff_t rext;
        ompi_datatype_type_extent(rdtype, &rext);

        if (MPI_IN_PLACE == sbuf) {
            _scount = rcounts[node_comm_rank];
            _sbuf = (char *) _rbuf + rext * displs[node_comm_rank];
            _sdtype = rdtype;
        }

        coll_info->rdtypes_ext[node_comm_rank][0] = rext;

        coll_info->sdtypes_contiguous[node_comm_rank][0] = 1;
        for (int i = 0; i < node_comm_size; ++i) {
            coll_info->rcounts[node_comm_rank][i] = rdtype_size * rcounts[i];
            coll_info->sdtypes_contiguous[node_comm_rank][0]
                &= ompi_datatype_is_contiguous_memory_layout(_sdtype, _scount)
                   & ompi_datatype_is_contiguous_memory_layout(rdtype,
                                                               rcounts[i]);
        }

        size_t collectivies_info_bytes_sended = 0;
        while (collectivies_info_bytes_sended
               < coll_info->one_rank_block_size) {
            int bytes_to_send = min(coll_info->one_rank_block_size
                                        - collectivies_info_bytes_sended,
                                    shm_data->mu_queue_fragment_size);
            SHARM_PROFILING_TIME_START(sharm_module, gatherv, push);
            int push = sharm_queue_push_contiguous(
                RESOLVE_COLLECTIVIES_DATA(sharm_module, node_comm_rank)
                    + collectivies_info_bytes_sended,
                bytes_to_send, root, -1, comm, sharm_module);
            SHARM_PROFILING_TIME_STOP(sharm_module, gatherv, push);
            collectivies_info_bytes_sended += push;
        }

        /*
         * Construct convertors to send messages.
         */
        for (int i = 0; i < node_comm_size; ++i) {
            recv_bytes_by_rank[i] = 0;
            char *recv_buff_ptr_for_rank = (char *) _rbuf + rext * displs[i];

            OBJ_CONSTRUCT(&root_convertors_by_rank[i], opal_convertor_t);
            if (i == node_comm_rank) {
                if (MPI_IN_PLACE != sbuf) {
                    ompi_datatype_sndrcv((char *) _sbuf, _scount, _sdtype,
                                         recv_buff_ptr_for_rank, rcounts[i],
                                         rdtype);
                }
                total_sizes_by_rank[i] = 0;
                continue;
            }

            if (OMPI_SUCCESS
                != (ret = opal_convertor_copy_and_prepare_for_recv(
                        ompi_mpi_local_convertor, &(rdtype->super), rcounts[i],
                        recv_buff_ptr_for_rank, 0,
                        &root_convertors_by_rank[i]))) {
                return ret;
            }
            total_sizes_by_rank[i] = coll_info->rcounts[node_comm_rank][i];
            total_size += total_sizes_by_rank[i];
        }

        /*
         * Recv messages from each process
         */
        size_t bytes_received = 0;
        while (bytes_received < total_size) {
            for (int i = 0; i < node_comm_size; ++i) {
                if (recv_bytes_by_rank[i] >= total_sizes_by_rank[i]) {
                    continue;
                }

                int pop = 0;
                pop = sharm_queue_pop(&(root_convertors_by_rank[i]), i, comm,
                                      sharm_module);
                bytes_received += pop;
                recv_bytes_by_rank[i] += pop;
            }
        }
        for (int i = 0; i < node_comm_size; ++i) {
            OBJ_DESTRUCT(&(root_convertors_by_rank[i]));
        }
    } else {
        opal_convertor_t convertor;
        OBJ_CONSTRUCT(&convertor, opal_convertor_t);

        size_t sdtype_size;
        ompi_datatype_type_size(sdtype, &sdtype_size);
        total_size = sdtype_size * scount;
        if (OMPI_SUCCESS
            != (ret = opal_convertor_copy_and_prepare_for_send(
                    ompi_mpi_local_convertor, &(_sdtype->super), _scount, _sbuf,
                    0, &convertor))) {
            return ret;
        }

        size_t collectivies_info_bytes_received = 0;
        while (collectivies_info_bytes_received
               < coll_info->one_rank_block_size) {
            SHARM_PROFILING_TIME_START(sharm_module, gatherv, pop);
            int pop = sharm_queue_pop_contiguous(
                RESOLVE_COLLECTIVIES_DATA(sharm_module, root)
                    + collectivies_info_bytes_received,
                root, comm, sharm_module);
            SHARM_PROFILING_TIME_STOP(sharm_module, gatherv, pop);
            collectivies_info_bytes_received += pop;
        }

        /*
         * Send message.
         */
        size_t bytes_sended = 0;
        while (bytes_sended < total_size) {
            int push = sharm_queue_push(&convertor,
                                        shm_data->mu_queue_fragment_size,
                                        node_comm_rank, root, comm,
                                        sharm_module);
            bytes_sended += push;
        }

        // Adjust slots counters for sync it.
        for (int i = 0; i < node_comm_size; ++i) {
            if (i == node_comm_rank || i == root) {
                continue;
            }
            adjust_queue_current_slot(i, 0,
                                      (coll_info->rcounts[root][i]
                                       + shm_data->mu_queue_fragment_size - 1)
                                          / shm_data->mu_queue_fragment_size,
                                      sharm_module);
        }
        OBJ_DESTRUCT(&convertor);
    }

    opal_atomic_wmb();

    OPAL_OUTPUT_VERBOSE(
        (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
         "coll:sharm:%d:gatherv_cico: (%d/%d/%s), root %d gatherv complete",
         SHARM_COLL(gatherv, sharm_module), node_comm_rank, node_comm_size,
         comm->c_name, root));

    return OMPI_SUCCESS;
}
