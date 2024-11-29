/** @file */

#include "coll_sharm.h"

/*
 * Shared-memory based queue 'pop' functions
 */

/*
 * External variables.
 */

extern int mca_coll_sharm_stream;
extern uint32_t mca_coll_sharm_one;

/*
 * Local variables.
 */

/**
 * @brief Pop data from queue.
 * @param[in] convertor Convertor for pack function.
 * @param[in] queue queue system number.
 * @param[in] subqueue subqueue in queue system number.
 * @param[in] comm mpi communicator.
 * @param[in] module sharm module structure.
 * @return bytes count.
 */
inline size_t _sharm_queue_pop(opal_convertor_t *convertor, int queue,
                               int subqueue, ompi_communicator_t *comm,
                               mca_coll_sharm_module_t *module)
{
    sharm_coll_data_t *shm_data = module->shared_memory_data;
    int comm_rank = ompi_comm_rank(comm);

    int current_slot = SHARM_CURRENT_SLOT_RESOLVE(module->shared_memory_data,
                                                  queue, subqueue);
    long int *ctrl = (long int *) SHARM_CTRL_RESOLVE(shm_data, queue, subqueue,
                                                     current_slot,
                                                     comm_rank);
    if (0 == *ctrl) {
        return 0;
    }

    size_t blocksize = *ctrl;

    struct iovec iov;
    iov.iov_base = SHARM_QUEUE_RESOLVE(shm_data, queue, subqueue, current_slot);
    iov.iov_len = blocksize;

    opal_convertor_unpack(convertor, &iov, &mca_coll_sharm_one, &blocksize);

    opal_atomic_wmb();

    // clean:
    // Notify about successfull receive data

    memset(ctrl, 0, shm_data->mu_cacheline_size);

    opal_atomic_wmb();

    increment_queue_current_slot(queue, subqueue, module);

    // exit:
    return blocksize;
}

/**
 * @brief Pop contigous data from queue.
 * @param[in] convertor Convertor for pack function.
 * Ignored if dataptr not NULL.
 * @param[in] dataptr pointer to data if datatype assumed contiguous.
 * @param[in] queue queue system number.
 * @param[in] subqueue subqueue in queue system number.
 * @param[in] comm mpi communicator.
 * @param[in] module sharm module structure.
 * @return bytes count.
 */
inline size_t _sharm_queue_pop_contiguous(void *dataptr, int queue,
                                          int subqueue,
                                          ompi_communicator_t *comm,
                                          mca_coll_sharm_module_t *module)
{
    sharm_coll_data_t *shm_data = module->shared_memory_data;
    int comm_rank = ompi_comm_rank(comm);

    int current_slot = SHARM_CURRENT_SLOT_RESOLVE(shm_data, queue, subqueue);
    long int *ctrl = (long int *) SHARM_CTRL_RESOLVE(shm_data, queue, subqueue,
                                                     current_slot,
                                                     comm_rank);
    if (0 == *ctrl) {
        return 0;
    }

    size_t blocksize = *ctrl;

    memcpy(dataptr,
           SHARM_QUEUE_RESOLVE(shm_data, queue, subqueue, current_slot),
           blocksize);

    opal_atomic_wmb();

    // clean:
    // Notify about successfull receive data

    memset(ctrl, 0, shm_data->mu_cacheline_size);

    opal_atomic_wmb();

    increment_queue_current_slot(queue, subqueue, module);

    // exit:
    return blocksize;
}
