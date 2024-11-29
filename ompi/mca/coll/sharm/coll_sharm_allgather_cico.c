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
 * @brief shared-memory based algorithm for Allgather using CICO approach.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_allgather_cico(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                         void *rbuf, int rcount, ompi_datatype_t *rdtype,
                         ompi_communicator_t *comm,
                         mca_coll_base_module_t *module)
{
    int ret = 0;
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;
    sharm_local_collectivies_data_t *coll_info = &(
        sharm_module->local_collectivies_info);
    size_t rtotal_size = 0;

    int comm_rank = ompi_comm_rank(comm);
    int comm_size = ompi_comm_size(comm);

    const char *_sbuf = sbuf;
    int _scount = scount;
    ompi_datatype_t *_sdtype = sdtype;

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:allgather_cico: (%d/%d/%s)",
                         SHARM_COLL(allgather, sharm_module), comm_rank,
                         comm_size, comm->c_name));

    char *recv_buff_ptr_for_rank = NULL;

    size_t *total_sizes_by_rank = (size_t *) sharm_module->local_op_memory_map;
    size_t *recv_bytes_by_rank = (size_t *) ((size_t *) total_sizes_by_rank
                                             + comm_size);
    opal_convertor_t *convertors_by_rank
        = (opal_convertor_t *) ((size_t *) recv_bytes_by_rank + comm_size);

    ptrdiff_t rext;
    ompi_datatype_type_extent(rdtype, &rext);

    /*
     * If sbuf is MPI_IN_PLACE - send data from rbuf.
     */
    if (MPI_IN_PLACE == sbuf) {
        _scount = rcount;
        _sbuf = (char *) rbuf + rext * rcount * comm_rank;
        _sdtype = rdtype;
    }

    size_t rdtype_size;
    ompi_datatype_type_size(rdtype, &rdtype_size);
    size_t sdtype_size;
    ompi_datatype_type_size(_sdtype, &sdtype_size);

    coll_info->rdtypes_ext[comm_rank][0] = rext;
    coll_info->scounts[comm_rank][0] = _scount * sdtype_size;
    coll_info->sdtypes_contiguous[comm_rank][0]
        = ompi_datatype_is_contiguous_memory_layout(_sdtype, _scount)
          & ompi_datatype_is_contiguous_memory_layout(rdtype,
                                                      rcount * comm_size);
    for (int i = 0; i < comm_size; ++i) {
        coll_info->rcounts[comm_rank][i] = rcount * rdtype_size;
    }

    /*
     * Construct convertors to send messages.
     */
    for (int i = 0; i < comm_size; ++i) {
        recv_bytes_by_rank[i] = 0;
        recv_buff_ptr_for_rank = (char *) rbuf + rext * rcount * i;

        OBJ_CONSTRUCT(&convertors_by_rank[i], opal_convertor_t);
        if (i == comm_rank) {
            if (MPI_IN_PLACE != sbuf) {
                ompi_datatype_sndrcv((char *) _sbuf, scount, sdtype,
                                     recv_buff_ptr_for_rank, rcount, rdtype);
            }
            recv_bytes_by_rank[i] = total_sizes_by_rank[i] = 0;
            continue;
        }
        if (OMPI_SUCCESS
            != (ret = opal_convertor_copy_and_prepare_for_recv(
                    ompi_mpi_local_convertor, &(rdtype->super), rcount,
                    recv_buff_ptr_for_rank, 0, &convertors_by_rank[i]))) {
            return ret;
        }

        total_sizes_by_rank[i] = coll_info->rcounts[comm_rank][i];
        rtotal_size += total_sizes_by_rank[i];
    }

    size_t stotal_size = 0;
    opal_convertor_t convertor;
    OBJ_CONSTRUCT(&convertor, opal_convertor_t);

    /*
     * Construct convertor to send message.
     */
    if (OMPI_SUCCESS
        != (ret = opal_convertor_copy_and_prepare_for_send(
                ompi_mpi_local_convertor, &(_sdtype->super), _scount, _sbuf, 0,
                &convertor))) {
        return ret;
    }

    stotal_size = coll_info->scounts[comm_rank][0];

    size_t bytes_received = 0;
    size_t bytes_sended = 0;

    /*
     * Data exchange.
     */
    while (bytes_received < rtotal_size || bytes_sended < stotal_size) {
        for (int i = 0; i < comm_size; ++i) {
            /*
             * Send data.
             */
            if (comm_rank == i && bytes_sended < stotal_size) {
                int push = 0;
                push = sharm_queue_push(&convertor,
                                        shm_data->mu_queue_fragment_size,
                                        comm_rank, -1, comm, sharm_module);

                bytes_sended += push;
            }
            /*
             * If all data received.
             */
            if (recv_bytes_by_rank[i] >= total_sizes_by_rank[i]) {
                continue;
            }

            /*
             * Receive data.
             */
            int pop = 0;
            pop = sharm_queue_pop(&(convertors_by_rank[i]), i, comm,
                                  sharm_module);
            bytes_received += pop;
            recv_bytes_by_rank[i] += pop;
        }
    }

    OBJ_DESTRUCT(&convertor);
    for (int i = 0; i < comm_size; ++i) {
        OBJ_DESTRUCT(&(convertors_by_rank[i]));
    }

    opal_atomic_wmb();

    OPAL_OUTPUT_VERBOSE(
        (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
         "coll:sharm:%d:allgather_cico: (%d/%d/%s), allgather complete",
         SHARM_COLL(allgather, sharm_module), comm_rank, comm_size,
         comm->c_name));
    return OMPI_SUCCESS;
}
