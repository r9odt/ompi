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
 * @brief shared-memory based algorithm for Bcast using CICO approach.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_bcast_cico(void *buff, int count, ompi_datatype_t *datatype, int root,
                     ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    int ret = 0;
    size_t ddt_size = 0;
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;
    // sharm_local_collectivies_data_t *coll_info = &(
    //     sharm_module->local_collectivies_info);

    opal_convertor_t convertor;
    OBJ_CONSTRUCT(&convertor, opal_convertor_t);

    int node_comm_rank = ompi_comm_rank(comm);
    int node_comm_size = ompi_comm_size(comm);

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:bcast_cico: (%d/%d/%s) root %d",
                         SHARM_COLL(bcast, sharm_module), node_comm_rank,
                         node_comm_size, comm->c_name, root));

    // void *memory_map = sharm_module->local_op_memory_map;
    // size_t *collectivies_info_bytes_received_by_rank = (size_t *) memory_map;

    ompi_datatype_type_size(datatype, &ddt_size);

    // We do not exchange this value because convertor is supoport send
    // contigous and recv non-contigous;
    uint8_t is_contiguous_dtype
        = ompi_datatype_is_contiguous_memory_layout(datatype, count);

    // coll_info->sdtypes_contiguous[node_comm_rank][0]
    //         = ompi_datatype_is_contiguous_memory_layout(datatype, count);

    /*
     * Exchange collectivies info.
     */
    // SHARM_PROFILING_TIME_START(sharm_module, bcast, collective_exchange);
    // memset(collectivies_info_bytes_received_by_rank, 0,
    //        node_comm_size * sizeof(size_t));
    // size_t collectivies_info_bytes_sended = 0;
    // size_t collectivies_info_bytes_received = 0;
    // size_t collectivies_info_bytes_total = (node_comm_size - 1)
    //                                        * coll_info->one_rank_block_size;
    // while (collectivies_info_bytes_sended < coll_info->one_rank_block_size
    //        || collectivies_info_bytes_received
    //               < collectivies_info_bytes_total) {
    //     for (int i = 0; i < node_comm_size; ++i) {
    //         if (i == node_comm_rank
    //             || collectivies_info_bytes_received_by_rank[i]
    //                    >= coll_info->one_rank_block_size) {
    //             continue;
    //         }
    //         SHARM_PROFILING_TIME_START(sharm_module, bcast, pop);
    //         int pop = sharm_queue_pop_contiguous(
    //             RESOLVE_COLLECTIVIES_DATA(sharm_module, i)
    //                 + collectivies_info_bytes_received_by_rank[i],
    //             i, comm, sharm_module);
    //         SHARM_PROFILING_TIME_STOP(sharm_module, bcast, pop);
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
    //     SHARM_PROFILING_TIME_START(sharm_module, bcast, push);
    //     int push = sharm_queue_push_contiguous(
    //         RESOLVE_COLLECTIVIES_DATA(sharm_module, node_comm_rank)
    //             + collectivies_info_bytes_sended,
    //         bytes_to_send, node_comm_rank, -1, comm, sharm_module);
    //     SHARM_PROFILING_TIME_STOP(sharm_module, bcast, push);
    //     collectivies_info_bytes_sended += push;
    // }

    // uint8_t is_contiguous_dtype = 1;
    // for (int i = 0; i < node_comm_size; ++i) {
    //     is_contiguous_dtype &= coll_info->sdtypes_contiguous[i][0];
    // }

    // SHARM_PROFILING_TIME_STOP(sharm_module, bcast, collective_exchange);

    size_t total_size = ddt_size * count;
    if (root == node_comm_rank) {
        if (0 == is_contiguous_dtype) {
            if (OMPI_SUCCESS
                != (ret = opal_convertor_copy_and_prepare_for_send(
                        ompi_mpi_local_convertor, &(datatype->super), count,
                        buff, 0, &convertor))) {
                return ret;
            }
        }

        size_t bytes_sended = 0;
        while (bytes_sended < total_size) {
            int push = 0;
            int bytes_to_send = min(total_size - bytes_sended,
                                    shm_data->mu_queue_fragment_size);
            if (is_contiguous_dtype) {
                SHARM_PROFILING_TIME_START(sharm_module, bcast, push);
                push = sharm_queue_push_contiguous(((char *) buff)
                                                       + bytes_sended,
                                                   bytes_to_send,
                                                   node_comm_rank, -1, comm,
                                                   sharm_module);
                SHARM_PROFILING_TIME_STOP(sharm_module, bcast, push);
            } else {
                SHARM_PROFILING_TIME_START(sharm_module, bcast, push);
                push = sharm_queue_push(&convertor,
                                        shm_data->mu_queue_fragment_size,
                                        node_comm_rank, -1, comm, sharm_module);
                SHARM_PROFILING_TIME_STOP(sharm_module, bcast, push);
            }
            bytes_sended += push;
        }
    } else {
        if (0 == is_contiguous_dtype) {
            if (OMPI_SUCCESS
                != (ret = opal_convertor_copy_and_prepare_for_recv(
                        ompi_mpi_local_convertor, &(datatype->super), count,
                        buff, 0, &convertor))) {
                return ret;
            }
        }
        size_t bytes_received = 0;
        while (bytes_received < total_size) {
            int pop = 0;
            if (is_contiguous_dtype) {
                SHARM_PROFILING_TIME_START(sharm_module, bcast, pop);
                pop = sharm_queue_pop_contiguous(((char *) buff)
                                                     + bytes_received,
                                                 root, comm, sharm_module);
                SHARM_PROFILING_TIME_STOP(sharm_module, bcast, pop);
            } else {
                SHARM_PROFILING_TIME_START(sharm_module, bcast, pop);
                pop = sharm_queue_pop(&convertor, root, comm, sharm_module);
                SHARM_PROFILING_TIME_STOP(sharm_module, bcast, pop);
            }
            bytes_received += pop;
        }
    }

    opal_atomic_wmb();

    OBJ_DESTRUCT(&convertor);

    OPAL_OUTPUT_VERBOSE(
        (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
         "coll:sharm:%d:bcast_cico: (%d/%d/%s), root %d bcast complete",
         SHARM_COLL(bcast, sharm_module), node_comm_rank, node_comm_size,
         comm->c_name, root));

    return OMPI_SUCCESS;
}
