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
 * Local functions.
 */

#define notify_block()                                                     \
    {                                                                      \
        long int *ctrl = (long int *) SHARM_CTRL_RESOLVE(shm_data, queue,  \
                                                         subqueue,         \
                                                         current_slot, 0); \
        for (int i = 0; i < node_comm_size; ++i) {                         \
            if ((notify < 0 && i != node_comm_rank) || i == notify         \
                || notify == node_comm_size)                               \
                *ctrl = blocksize;                                         \
            opal_atomic_wmb();                                             \
            ctrl = (long int *) (((unsigned char *) ctrl)                  \
                                 + shm_data->mu_cacheline_size);           \
        }                                                                  \
    }

/*
* If we notify all without self or specific rank or all ranks. *ctrl =
blocksize; if ((notify < 0 && i != node_comm_rank) || i == notify
            || notify == node_comm_size)

* else if we notify spec process and want notify other to skip this and set
* *ctrl = -1;
else if (i != notify && 0 < notify && notify < node_comm_size
&& notifyFragmentNotFor)
*/

/**
 * @brief Push block with blocksize into queue.
 * @param[in] convertor Convertor for pack function.
 * Ignored if dataptr not NULL.
 * @param[in] blocksize size of block to push into queue.
 * @param[in] queue queue system number.
 * @param[in] subqueue subqueue in queue system number.
 * @param[in] notify
 * - if notify == rank: notify rank.
 * - if notify == -1: notify all without self.
 * - if notify == comm_size: notify all.
 * @param[in] comm mpi communicator.
 * @param[in] module sharm module structure.
 * @param[in] notifyFragmentNotFor if 1, processes which should be excluded
 * from this fragment will be notified with -1 value.
 * NOTE: DISABLED
 * @return pushed bytes count.
 */
inline size_t _sharm_queue_push(opal_convertor_t *convertor, size_t blocksize,
                                int queue, int subqueue, int notify,
                                ompi_communicator_t *comm,
                                mca_coll_sharm_module_t *module)
{
    sharm_coll_data_t *shm_data = module->shared_memory_data;

    int node_comm_rank = ompi_comm_rank(comm);
    int node_comm_size = ompi_comm_size(comm);

    // TODO: OPAL_ENABLE_DEBUG
    // if (OPAL_UNLIKELY(!notify_showed && node_comm_rank != queue)) {
    //     notify_showed = 1;
    //     opal_output(
    //         mca_coll_sharm_stream,
    //         "coll:sharm:queue:push (%d/%d/%s) trying to use queue system %d.
    //         " "Using queue system which number is not equal with rank may
    //         cause " "out of sync.", node_comm_rank, node_comm_size,
    //         comm->c_name, queue);
    // }
    int current_slot = SHARM_CURRENT_SLOT_RESOLVE(shm_data, queue, subqueue);

    /*
     * Check slot become free.
     */
    if (check_controls_is_zero(queue, subqueue, current_slot, comm, module)
        != 0) {
        blocksize = 0;
        goto exit;
    }

    struct iovec iov;
    iov.iov_len = blocksize;
    iov.iov_base = SHARM_QUEUE_RESOLVE(shm_data, queue, subqueue, current_slot);

    opal_convertor_pack(convertor, &iov, &mca_coll_sharm_one, &blocksize);
    opal_atomic_wmb();

    /*
     * Notify receivers.
     */

    notify_block();

    increment_queue_current_slot(queue, subqueue, module);

exit:
    return blocksize;
}

/**
 * @brief Push contiguous block with blocksize into queue.
 * @param[in] dataptr data pointer to data.
 * @param[in] blocksize size of block to push into queue.
 * @param[in] queue queue system number.
 * @param[in] subqueue subqueue in queue system number.
 * @param[in] notify
 * - if notify == rank: notify rank.
 * - if notify == -1: notify all without self.
 * - if notify == comm_size: notify all.
 * @param[in] comm mpi communicator.
 * @param[in] module sharm module structure.
 * @param[in] notifyFragmentNotFor if 1, processes which should be excluded
 * from this fragment will be notified with -1 value.
 * NOTE: DISABLED
 * @return pushed bytes count.
 */
inline size_t _sharm_queue_push_contiguous(const void *dataptr,
                                           size_t blocksize, int queue,
                                           int subqueue, int notify,
                                           ompi_communicator_t *comm,
                                           mca_coll_sharm_module_t *module)
{
    sharm_coll_data_t *shm_data = module->shared_memory_data;

    int node_comm_rank = ompi_comm_rank(comm);
    int node_comm_size = ompi_comm_size(comm);

    // TODO: OPAL_ENABLE_DEBUG
    // if (OPAL_UNLIKELY(!notify_showed && node_comm_rank != queue)) {
    //     notify_showed = 1;
    //     opal_output(
    //         mca_coll_sharm_stream,
    //         "coll:sharm:queue:push (%d/%d/%s) trying to use queue system %d.
    //         " "Using queue system which number is not equal with rank may
    //         cause " "out of sync.", node_comm_rank, node_comm_size,
    //         comm->c_name, queue);
    // }
    int current_slot = SHARM_CURRENT_SLOT_RESOLVE(shm_data, queue, subqueue);

    /*
     * Check slot become free.
     */
    if (check_controls_is_zero(queue, subqueue, current_slot, comm, module)
        != 0) {
        blocksize = 0;
        goto exit;
    }

    memcpy(SHARM_QUEUE_RESOLVE(shm_data, queue, subqueue, current_slot),
           dataptr, blocksize);

    opal_atomic_wmb();

    /*
     * Notify receivers.
     */

    notify_block();

    increment_queue_current_slot(queue, subqueue, module);

exit:
    return blocksize;
}
