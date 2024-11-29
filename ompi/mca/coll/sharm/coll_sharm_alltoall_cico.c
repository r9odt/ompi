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
 * @brief shared-memory based algorithm for Alltoall using CICO approach.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_alltoall_cico(const void *sbuf, int scount,
                        struct ompi_datatype_t *sdtype, void *rbuf, int rcount,
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

    int comm_rank = ompi_comm_rank(comm);
    int comm_size = ompi_comm_size(comm);

    const char *_sbuf = sbuf;
    int _scount = scount;
    ompi_datatype_t *_sdtype = sdtype;

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:alltoall_cico_pairwise: (%d/%d/%s)",
                         SHARM_COLL(alltoall, sharm_module), comm_rank,
                         comm_size, comm->c_name));

    if (MPI_IN_PLACE == sbuf) {
        _scount = rcount;
        _sbuf = (char *) rbuf;
        _sdtype = rdtype;
    }

    /*
     * rtotal_sizes_by_rank - sizeof(size_t) * comm_size
     * stotal_sizes_by_rank - sizeof(size_t) * comm_size
     * recv_bytes_by_rank - sizeof(size_t) * comm_size
     * rconvertors_by_rank - sizeof(opal_convertor_t) * comm_size
     * sconvertors_by_rank - sizeof(opal_convertor_t) * comm_size
     * send_bytes_by_rank - sizeof(size_t) * comm_size
     */

    void *memory_map = sharm_module->local_op_memory_map;
    size_t *rtotal_sizes_by_rank = (size_t *) memory_map;
    size_t *stotal_sizes_by_rank = (size_t *) ((size_t *) rtotal_sizes_by_rank
                                               + comm_size);
    size_t *recv_bytes_by_rank = (size_t *) ((size_t *) stotal_sizes_by_rank
                                             + comm_size);
    size_t *send_bytes_by_rank = (size_t *) ((size_t *) recv_bytes_by_rank
                                             + comm_size);
    size_t *collectivies_info_bytes_received_by_rank
        = (size_t *) ((size_t *) send_bytes_by_rank + comm_size);
    char **recv_buff_ptr_for_rank
        = (char **) ((size_t *) collectivies_info_bytes_received_by_rank
                     + comm_size);
    char **send_buff_ptr_for_rank = (char **) ((char **) recv_buff_ptr_for_rank
                                               + comm_size);
    opal_convertor_t *rconvertors_by_rank
        = (opal_convertor_t *) ((char **) send_buff_ptr_for_rank
                                + comm_size);
    opal_convertor_t *sconvertors_by_rank
        = (opal_convertor_t *) ((opal_convertor_t *) rconvertors_by_rank
                                + comm_size);

    ompi_datatype_type_size(_sdtype, &sdtype_size);
    ompi_datatype_type_size(rdtype, &rdtype_size);

    /*
     * If datatype is not contigous - use pack/unpack methods.
     */
    ptrdiff_t rext;
    ompi_datatype_type_extent(rdtype, &rext);
    ptrdiff_t sext;
    ompi_datatype_type_extent(_sdtype, &sext);
    coll_info->sdtypes_ext[comm_rank][0] = sext;
    coll_info->rdtypes_ext[comm_rank][0] = rext;

    uint8_t is_contiguous_dtype = coll_info
                                      ->sdtypes_contiguous[comm_rank][0]
        = ompi_datatype_is_contiguous_memory_layout(_sdtype,
                                                    _scount * comm_size)
          & ompi_datatype_is_contiguous_memory_layout(rdtype,
                                                      rcount * comm_size);

    /*
     * Exchange collectivies info.
     */
    // memset(collectivies_info_bytes_received_by_rank, 0,
    //        comm_size * sizeof(size_t));
    // size_t collectivies_info_bytes_sended = 0;
    // size_t collectivies_info_bytes_received = 0;
    // size_t collectivies_info_bytes_total = (comm_size - 1)
    //                                        * coll_info->one_rank_block_size;
    // while (collectivies_info_bytes_sended < coll_info->one_rank_block_size
    //        || collectivies_info_bytes_received
    //               < collectivies_info_bytes_total) {
    //     for (int i = 0; i < comm_size; ++i) {
    //         if (i == comm_rank
    //             || collectivies_info_bytes_received_by_rank[i]
    //                    >= coll_info->one_rank_block_size) {
    //             continue;
    //         }
    //         int pop = sharm_queue_pop_contiguous(
    //             RESOLVE_COLLECTIVIES_DATA(sharm_module, i)
    //                 + collectivies_info_bytes_received_by_rank[i],
    //             i, comm, sharm_module);
    //         collectivies_info_bytes_received_by_rank[i] += pop;
    //         collectivies_info_bytes_received += pop;
    //     }

    //     if (collectivies_info_bytes_sended >= coll_info->one_rank_block_size)
    //     {
    //         continue;
    //     }

    //     int bytes_to_send = min(coll_info->one_rank_block_size
    //                                 - collectivies_info_bytes_sended,
    //                             shm_data->mu_queue_fragment_size);
    //     SHARM_PROFILING_TIME_START(sharm_module, alltoall, push);
    //     int push = sharm_queue_push_contiguous(
    //         RESOLVE_COLLECTIVIES_DATA(sharm_module, comm_rank)
    //             + collectivies_info_bytes_sended,
    //         bytes_to_send, comm_rank, -1, comm, sharm_module);
    //     SHARM_PROFILING_TIME_STOP(sharm_module, alltoall, push);
    //     collectivies_info_bytes_sended += push;
    // }

    // uint8_t is_contiguous_dtype = 1;
    // for (int i = 0; i < comm_size; ++i) {
    //     is_contiguous_dtype &= coll_info->sdtypes_contiguous[i][0];
    // }

    /*
     * Construct convertors to send messages.
     */
    for (int i = 0; i < comm_size; ++i) {
        recv_bytes_by_rank[i] = 0;
        send_bytes_by_rank[i] = 0;
        recv_buff_ptr_for_rank[i] = (char *) rbuf + rext * rcount * i;
        send_buff_ptr_for_rank[i] = (char *) _sbuf + sext * _scount * i;

        OBJ_CONSTRUCT(&sconvertors_by_rank[i], opal_convertor_t);
        OBJ_CONSTRUCT(&rconvertors_by_rank[i], opal_convertor_t);
        if (i == comm_rank) {
            if (MPI_IN_PLACE != sbuf) {
                ompi_datatype_sndrcv(send_buff_ptr_for_rank[i], (_scount),
                                     _sdtype, recv_buff_ptr_for_rank[i], rcount,
                                     rdtype);
            }
            stotal_sizes_by_rank[i] = 0;
            rtotal_sizes_by_rank[i] = 0;
            continue;
        }

        rtotal_sizes_by_rank[i] = rdtype_size * rcount;
        stotal_sizes_by_rank[i] = sdtype_size * _scount;
        if (0 == is_contiguous_dtype) {
            if (OMPI_SUCCESS
                != (ret = opal_convertor_copy_and_prepare_for_recv(
                        ompi_mpi_local_convertor, &(rdtype->super), rcount,
                        recv_buff_ptr_for_rank[i], 0,
                        &rconvertors_by_rank[i]))) {
                return ret;
            }

            if (OMPI_SUCCESS
                != (ret = opal_convertor_copy_and_prepare_for_send(
                        ompi_mpi_local_convertor, &(_sdtype->super), (_scount),
                        send_buff_ptr_for_rank[i], 0,
                        &sconvertors_by_rank[i]))) {
                return ret;
            }
        }
        rtotal_size += rtotal_sizes_by_rank[i];
        stotal_size += stotal_sizes_by_rank[i];
    }

    size_t bytes_received = 0;
    size_t bytes_sended = 0;

    /*
     * Data exchange.
     */
    while (bytes_received < rtotal_size || bytes_sended < stotal_size) {
        for (int i = 1; i < comm_size; ++i) {
            int sendto = (comm_rank + i) % comm_size;
            int recvfrom = (comm_rank + comm_size - i)
                           % comm_size;
            if (send_bytes_by_rank[sendto] < stotal_sizes_by_rank[sendto]) {
                int push = 0;

                if (is_contiguous_dtype) {
                    int bytes_to_send = min(stotal_sizes_by_rank[sendto]
                                                - send_bytes_by_rank[sendto],
                                            shm_data->mu_queue_fragment_size);
                    push = sharm_queue_push_to_subqueue_contiguous(
                        send_buff_ptr_for_rank[sendto]
                            + send_bytes_by_rank[sendto],
                        bytes_to_send, comm_rank, sendto, sendto, comm,
                        sharm_module);
                } else {
                    push = sharm_queue_push_to_subqueue(
                        &(sconvertors_by_rank[sendto]),
                        shm_data->mu_queue_fragment_size, comm_rank,
                        sendto, sendto, comm, sharm_module);
                }
                bytes_sended += push;
                send_bytes_by_rank[sendto] += push;
            }
            if (recv_bytes_by_rank[recvfrom]
                >= rtotal_sizes_by_rank[recvfrom]) {
                continue;
            }

            int pop = 0;
            if (is_contiguous_dtype) {
                pop = sharm_queue_pop_from_subqueue_contiguous(
                    recv_buff_ptr_for_rank[recvfrom]
                        + recv_bytes_by_rank[recvfrom],
                    recvfrom, comm_rank, comm, sharm_module);
            } else {
                pop = sharm_queue_pop_from_subqueue(
                    &(rconvertors_by_rank[recvfrom]), recvfrom, comm_rank,
                    comm, sharm_module);
            }
            bytes_received += pop;
            recv_bytes_by_rank[recvfrom] += pop;
        }
    }

    for (int i = 0; i < comm_size; ++i) {
        OBJ_DESTRUCT(&(rconvertors_by_rank[i]));
        OBJ_DESTRUCT(&(sconvertors_by_rank[i]));
    }

    // Adjust slots counters for sync it.
    for (int i = 0; i < comm_size; ++i) {
        if (i == comm_rank)
            continue;
        for (int j = 0; j < comm_size; ++j) {
            if (j == comm_rank || i == j) {
                continue;
            }
            adjust_queue_current_slot(i, j,
                                      (stotal_sizes_by_rank[j]
                                       + shm_data->mu_queue_fragment_size - 1)
                                          / shm_data->mu_queue_fragment_size,
                                      sharm_module);
        }
    }

    opal_atomic_wmb();

    OPAL_OUTPUT_VERBOSE(
        (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
         "coll:sharm:%d:alltoall_cico_pairwise: (%d/%d/%s), alltoall complete",
         SHARM_COLL(alltoall, sharm_module), comm_rank, comm_size,
         comm->c_name));
    return OMPI_SUCCESS;
}
