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
 * @brief shared-memory based algorithm for Alltoallv using CICO approach.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_alltoallv_cico(const void *sbuf, const int *scounts,
                         const int *sdispls, struct ompi_datatype_t *sdtype,
                         void *rbuf, const int *rcounts, const int *rdispls,
                         struct ompi_datatype_t *rdtype,
                         struct ompi_communicator_t *comm,
                         mca_coll_base_module_t *module)
{
    int ret = 0;
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;
    sharm_local_collectivies_data_t *coll_info = &(
        sharm_module->local_collectivies_info);
    size_t sdtype_size;
    size_t rdtype_size;
    size_t rtotal_size = 0;
    size_t stotal_size = 0;

    int node_comm_rank = ompi_comm_rank(comm);
    int node_comm_size = ompi_comm_size(comm);

    const char *_sbuf = (char *) sbuf;
    const int *_scounts = scounts;
    ompi_datatype_t *_sdtype = sdtype;
    const int *_sdispls = sdispls;

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:alltoallv_cico: (%d/%d/%s)",
                         SHARM_COLL(alltoallv, sharm_module), node_comm_rank,
                         node_comm_size, comm->c_name));

    if (MPI_IN_PLACE == sbuf) {
        _scounts = rcounts;
        _sbuf = (char *) rbuf;
        _sdtype = rdtype;
        _sdispls = rdispls;
    }

    /*
     * rtotal_sizes_by_rank - sizeof(size_t) * node_comm_size
     * stotal_sizes_by_rank - sizeof(size_t) * node_comm_size
     * recv_bytes_by_rank - sizeof(size_t) * node_comm_size
     * rconvertors_by_rank - sizeof(opal_convertor_t) * node_comm_size
     * send_bytes_by_rank - sizeof(size_t) * node_comm_size
     * sconvertors_by_rank - sizeof(opal_convertor_t) * node_comm_size
     * stotal_sizes_by_rank_for_all_ranks - sizeof(size_t) * node_comm_size *
     * node_comm_size
     */

    void *memory_map = sharm_module->local_op_memory_map;
    size_t *rtotal_sizes_by_rank = (size_t *) memory_map;
    size_t *stotal_sizes_by_rank = (size_t *) ((size_t *) rtotal_sizes_by_rank
                                               + node_comm_size);
    size_t *send_bytes_by_rank = (size_t *) ((size_t *) stotal_sizes_by_rank
                                             + node_comm_size);
    size_t *recv_bytes_by_rank = (size_t *) ((size_t *) send_bytes_by_rank
                                             + node_comm_size);
    size_t *collectivies_info_bytes_received_by_rank
        = (size_t *) ((size_t *) recv_bytes_by_rank + node_comm_size);
    opal_convertor_t *rconvertors_by_rank
        = (opal_convertor_t *) ((size_t *)
                                    collectivies_info_bytes_received_by_rank
                                + node_comm_size);
    opal_convertor_t *sconvertors_by_rank
        = (opal_convertor_t *) ((opal_convertor_t *) rconvertors_by_rank
                                + node_comm_size);

    ompi_datatype_type_size(_sdtype, &sdtype_size);
    ompi_datatype_type_size(rdtype, &rdtype_size);
    ptrdiff_t rext;
    ompi_datatype_type_extent(rdtype, &rext);
    ptrdiff_t sext;
    ompi_datatype_type_extent(_sdtype, &sext);
    coll_info->sdtypes_ext[node_comm_rank][0] = sext;
    coll_info->rdtypes_ext[node_comm_rank][0] = rext;

    coll_info->sdtypes_contiguous[node_comm_rank][0] = 1;
    for (int i = 0; i < node_comm_size; ++i) {
        coll_info->sdtypes_contiguous[node_comm_rank][0]
            &= ompi_datatype_is_contiguous_memory_layout(_sdtype, _scounts[i])
               & ompi_datatype_is_contiguous_memory_layout(rdtype, rcounts[i]);
        coll_info->sbuf[node_comm_rank][i] = (ptrdiff_t) (_sbuf
                                                          + sext * _sdispls[i]);
        coll_info->scounts[node_comm_rank][i] = _scounts[i] * sdtype_size;
        coll_info->rcounts[node_comm_rank][i] = rcounts[i] * rdtype_size;
    }

    /*
     * Exchange collectivies info.
     */
    memset(collectivies_info_bytes_received_by_rank, 0,
           node_comm_size * sizeof(size_t));
    size_t collectivies_info_bytes_sended = 0;
    size_t collectivies_info_bytes_received = 0;
    size_t collectivies_info_bytes_total = (node_comm_size - 1)
                                           * coll_info->one_rank_block_size;
    while (collectivies_info_bytes_sended < coll_info->one_rank_block_size
           || collectivies_info_bytes_received
                  < collectivies_info_bytes_total) {
        for (int i = 0; i < node_comm_size; ++i) {
            if (i == node_comm_rank
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
        SHARM_PROFILING_TIME_START(sharm_module, alltoallv, push);
        int push = sharm_queue_push_contiguous(
            RESOLVE_COLLECTIVIES_DATA(sharm_module, node_comm_rank)
                + collectivies_info_bytes_sended,
            bytes_to_send, node_comm_rank, -1, comm, sharm_module);
        SHARM_PROFILING_TIME_STOP(sharm_module, alltoallv, push);
        collectivies_info_bytes_sended += push;
    }

    // uint8_t is_contiguous_dtype = 1;
    // for (int i = 0; i < node_comm_size; ++i) {
    //     if (coll_info->sdtypes_contiguous[i][0] == 0) {
    //         is_contiguous_dtype = 0;
    //         break;
    //     }
    // }

    /*
     * Construct convertors to send messages.
     */
    for (int i = 0; i < node_comm_size; ++i) {
        recv_bytes_by_rank[i] = 0;
        send_bytes_by_rank[i] = 0;
        char *recv_buff_ptr_for_rank = (char *) rbuf + rext * rdispls[i];
        char *send_buff_ptr_for_rank = (char *) _sbuf + sext * _sdispls[i];

        OBJ_CONSTRUCT(&sconvertors_by_rank[i], opal_convertor_t);
        OBJ_CONSTRUCT(&rconvertors_by_rank[i], opal_convertor_t);
        if (OPAL_UNLIKELY(i == node_comm_rank)) {
            if (MPI_IN_PLACE != sbuf) {
                ompi_datatype_sndrcv(send_buff_ptr_for_rank, _scounts[i],
                                     _sdtype, recv_buff_ptr_for_rank,
                                     rcounts[i], rdtype);
            }
            rtotal_sizes_by_rank[i] = 0;
            stotal_sizes_by_rank[i] = 0;
            continue;
        }

        if (OMPI_SUCCESS
            != (ret = opal_convertor_copy_and_prepare_for_recv(
                    ompi_mpi_local_convertor, &(rdtype->super), rcounts[i],
                    recv_buff_ptr_for_rank, 0, &rconvertors_by_rank[i]))) {
            return ret;
        }

        if (OMPI_SUCCESS
            != (ret = opal_convertor_copy_and_prepare_for_send(
                    ompi_mpi_local_convertor, &(_sdtype->super), _scounts[i],
                    send_buff_ptr_for_rank, 0, &sconvertors_by_rank[i]))) {
            return ret;
        }

        rtotal_sizes_by_rank[i] = coll_info->rcounts[node_comm_rank][i];
        stotal_sizes_by_rank[i] = coll_info->scounts[node_comm_rank][i];
        rtotal_size += rtotal_sizes_by_rank[i];
        stotal_size += stotal_sizes_by_rank[i];
    }

    size_t bytes_received = 0;
    size_t bytes_sended = 0;

    /*
     * Data exchange.
     */
    while (bytes_received < rtotal_size || bytes_sended < stotal_size) {
        // Pairwise
        for (int i = 1; i < node_comm_size; ++i) {
            int sendto = (node_comm_rank + i) % node_comm_size;
            int recvfrom = (node_comm_rank + node_comm_size - i)
                           % node_comm_size;
            if (send_bytes_by_rank[sendto] < stotal_sizes_by_rank[sendto]) {
                int push = 0;

                SHARM_PROFILING_TIME_START(sharm_module, alltoallv, push);
                push = sharm_queue_push_to_subqueue(
                    &(sconvertors_by_rank[sendto]),
                    shm_data->mu_queue_fragment_size, node_comm_rank, sendto,
                    sendto, comm, sharm_module);
                SHARM_PROFILING_TIME_STOP(sharm_module, alltoallv, push);

                bytes_sended += push;
                send_bytes_by_rank[sendto] += push;

                /*
                 * TODO: For support MPI_IN_PLACE ensure that we sended more
                 * bytes than will receive. For cases when recvbytes !=
                 * sendbytes we check fragment size directly.
                 * Also it is small optimization for queue_pop without code jump
                 */
                if (send_bytes_by_rank[recvfrom]
                    < recv_bytes_by_rank[recvfrom]
                          + sharm_queue_get_subqueue_ctrl(recvfrom,
                                                          node_comm_rank, comm,
                                                          sharm_module)) {
                    continue;
                }
            }

            if (recv_bytes_by_rank[recvfrom]
                >= rtotal_sizes_by_rank[recvfrom]) {
                continue;
            }

            int pop = 0;

            SHARM_PROFILING_TIME_START(sharm_module, alltoallv, pop);
            pop = sharm_queue_pop_from_subqueue(
                &(rconvertors_by_rank[recvfrom]), recvfrom, node_comm_rank,
                comm, sharm_module);
            SHARM_PROFILING_TIME_STOP(sharm_module, alltoallv, pop);

            bytes_received += pop;
            recv_bytes_by_rank[recvfrom] += pop;
        }
    }

    for (int i = 0; i < node_comm_size; ++i) {
        OBJ_DESTRUCT(&(rconvertors_by_rank[i]));
        OBJ_DESTRUCT(&(sconvertors_by_rank[i]));
    }

    // Adjust slots counters for sync it.
    for (int i = 0; i < node_comm_size; ++i) {
        if (OPAL_UNLIKELY(i == node_comm_rank))
            continue;
        for (int j = 0; j < node_comm_size; ++j) {
            if (OPAL_UNLIKELY(j == node_comm_rank || i == j)) {
                continue;
            }

            adjust_queue_current_slot(i, j,
                                      (coll_info->scounts[i][j]
                                       + shm_data->mu_queue_fragment_size - 1)
                                          / shm_data->mu_queue_fragment_size,
                                      sharm_module);
        }
    }

    opal_atomic_wmb();

    OPAL_OUTPUT_VERBOSE(
        (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
         "coll:sharm:%d:alltoallv_cico: (%d/%d/%s), alltoallv complete",
         SHARM_COLL(alltoallv, sharm_module), node_comm_rank, node_comm_size,
         comm->c_name));

    return OMPI_SUCCESS;
}
