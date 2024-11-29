/** @file */

#include "coll_sharm.h"

/*
 * Shared-memory based queue functions
 */

/*
 * External variables.
 */

extern int mca_coll_sharm_stream;
/*
 * Local variables.
 */
uint32_t mca_coll_sharm_one = 1;

/*
 * Local functions.
 */

/**
 * @brief Adjust queue.s current_slot for specified value.
 * @param[in] queue queue system number.
 * @param[in] subqueue subqueue in queue system number.
 * @param[in] value value to adjust.
 * @param[in] module sharm module structure.
 */
// inline void adjust_queue_current_slot(int queue, int subqueue, int value,
//                                       mca_coll_sharm_module_t *module)
// {
//     sharm_coll_data_t *shm_data = module->shared_memory_data;
//     shm_data->current_queue_slots[queue][subqueue]
//         = (shm_data->current_queue_slots[queue][subqueue] + value)
//           % shm_data->mu_queue_nfrags;
// }

/**
 * @brief
 * @param[in] queue queue system number.
 * @param[in] subqueue subqueue in queue system number.
 * @param[in] slot queue slot to check.
 * @param[in] comm mpi communicator.
 * @param[in] module sharm module structure.
 * @return 0 if all control flags is zero else 1
 */
int check_controls_is_zero(int queue, int subqueue, int slot,
                           ompi_communicator_t *comm,
                           mca_coll_sharm_module_t *module)
{
    sharm_coll_data_t *shm_data = module->shared_memory_data;
    long int *ctrl = (long int *) SHARM_CTRL_RESOLVE(shm_data, queue, subqueue,
                                                     slot, 0);
    int i = ompi_comm_size(comm);

    while (i--) {
        if (OPAL_UNLIKELY(*ctrl != 0)) {
            return 1;
        }
        ctrl = (long int *) (((unsigned char *) ctrl)
                             + shm_data->mu_cacheline_size);
    }

    return 0;
}

/**
 * @brief Get control element value for queue and current rank.
 * @param[in] queue queue system number.
 * @param[in] subqueue subqueue in queue system number.
 * @param[in] comm mpi communicator.
 * @param[in] module sharm module structure.
 * @return bytes count.
 */
inline long int _sharm_queue_get_ctrl(int queue, int subqueue,
                                      ompi_communicator_t *comm,
                                      mca_coll_sharm_module_t *module)
{
    sharm_coll_data_t *shm_data = module->shared_memory_data;
    int comm_rank = ompi_comm_rank(comm);
    int current_slot = SHARM_CURRENT_SLOT_RESOLVE(shm_data, queue, subqueue);
    long int *ctrl = (long int *) SHARM_CTRL_RESOLVE(shm_data, queue, subqueue,
                                                     current_slot,
                                                     comm_rank);
    return *ctrl;
}

/**
 * @brief Get queue pointer.
 * @param[in] queue queue system number.
 * @param[in] subqueue subqueue in queue system number.
 * @param[in] comm mpi communicator.
 * @param[in] module sharm module structure.
 * @return bytes count.
 */
inline void *_sharm_queue_get_ptr(int queue, int subqueue,
                                  ompi_communicator_t *comm,
                                  mca_coll_sharm_module_t *module)
{
    sharm_coll_data_t *shm_data = module->shared_memory_data;
    int current_slot = SHARM_CURRENT_SLOT_RESOLVE(shm_data, queue, subqueue);

    return SHARM_QUEUE_RESOLVE(shm_data, queue, subqueue, current_slot);
}

/**
 * @brief Clear control element value for queue.
 * Increment current slot counter.
 * @param[in] queue queue system number.
 * @param[in] subqueue subqueue in queue system number.
 * @param[in] comm mpi communicator.
 * @param[in] module sharm module structure.
 */
inline void _sharm_queue_clear_ctrl(int queue, int subqueue,
                                    ompi_communicator_t *comm,
                                    mca_coll_sharm_module_t *module)
{
    sharm_coll_data_t *shm_data = module->shared_memory_data;
    int comm_rank = ompi_comm_rank(comm);
    int current_slot = SHARM_CURRENT_SLOT_RESOLVE(shm_data, queue, subqueue);
    long int *ctrl = (long int *) SHARM_CTRL_RESOLVE(shm_data, queue, subqueue,
                                                     current_slot,
                                                     comm_rank);
    memset(ctrl, 0, shm_data->mu_cacheline_size);
    opal_atomic_wmb();
    increment_queue_current_slot(queue, subqueue, module);
}
