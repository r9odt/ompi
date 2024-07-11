/** @file */

#ifndef MCA_COLL_SHARM_QUEUE_H
#define MCA_COLL_SHARM_QUEUE_H

extern uint32_t mca_coll_sharm_one;

#define wait_queue_func(var, func)      SHARM_SPIN_CONDITION((var = func) > 0)
#define wait_queue_func_no_return(func) SHARM_SPIN_CONDITION(func > 0)

/**
 * @brief Resolve pointer to queue.
 * @param[in] shm_data Pointer to shared memory structure.
 * @param[in] q queue system number.
 * @param[in] sq subqueue in queue system number.
 * @param[in] slot queue slot to check.
 * @return pointer to queue.
 */
#define SHARM_QUEUE_RESOLVE(shm_data, q, sq, slot)                   \
    (shm_data->shm_queue[q] + sq * shm_data->mu_queue_one_queue_size \
     + slot * shm_data->mu_queue_fragment_size)

/**
 * @brief Resolve pointer to queue.s control element for process.
 * @param[in] shm_data Pointer to shared memory structure.
 * @param[in] q queue system number.
 * @param[in] sq subqueue in queue system number.
 * @param[in] slot queue slot to check.
 * @param[in] proc process number.
 * @return pointer to control element.
 */
#define SHARM_CTRL_RESOLVE(shm_data, q, sq, slot, proc)                  \
    (shm_data->shm_control[q] + sq * shm_data->mu_control_one_queue_size \
     + slot * shm_data->mu_control_fragment_size                         \
     + proc * shm_data->mu_cacheline_size)

/**
 * @brief Resolve pointer to queue.s current slot value.
 * @param[in] shm_data Pointer to shared memory structure.
 * @param[in] q queue system number.
 * @param[in] sq subqueue in queue system number.
 * @return pointer to control element.
 */
#define SHARM_CURRENT_SLOT_RESOLVE(shm_data, q, sq) \
    (shm_data->current_queue_slots[q * shm_data->nproc + sq])

/**
 * @brief Adjust queue.s current_slot for specified value.
 * @param[in] queue queue system number.
 * @param[in] subqueue subqueue in queue system number.
 * @param[in] value value to adjust.
 * @param[in] module sharm module structure.
 */
#define adjust_queue_current_slot(queue, subqueue, value, module)            \
    {                                                                        \
        SHARM_CURRENT_SLOT_RESOLVE(module->shared_memory_data, queue,        \
                                   subqueue)                                 \
            = (SHARM_CURRENT_SLOT_RESOLVE(module->shared_memory_data, queue, \
                                          subqueue)                          \
               + value)                                                      \
              % module->shared_memory_data->mu_queue_nfrags;                 \
    }

#define increment_queue_current_slot(q, sq, m) \
    adjust_queue_current_slot(q, sq, 1, m)

/**
 * @brief Push to subqueue 0.
 */
#define sharm_queue_push(dataptr, blocksize, q, notify, comm, module) \
    _sharm_queue_push(dataptr, blocksize, q, 0, notify, comm, module)

/**
 * @brief Push to subqueue 0. Only contiguous types.
 */
#define sharm_queue_push_contiguous(dataptr, blocksize, q, notify, comm, \
                                    module)                              \
    _sharm_queue_push_contiguous(dataptr, blocksize, q, 0, notify, comm, module)

/**
 * @brief Push to subqueue with specified number.
 */
#define sharm_queue_push_to_subqueue(dataptr, blocksize, q, sq, notify, comm, \
                                     module)                                  \
    _sharm_queue_push(dataptr, blocksize, q, sq, notify, comm, module)

/**
 * @brief Push to subqueue with specified number. Only contiguous types.
 */
#define sharm_queue_push_to_subqueue_contiguous(dataptr, blocksize, q, sq, \
                                                notify, comm, module)      \
    _sharm_queue_push_contiguous(dataptr, blocksize, q, sq, notify, comm,  \
                                 module)

/**
 * @brief Pop from subqueue 0.
 */
#define sharm_queue_pop(dataptr, q, comm, module) \
    _sharm_queue_pop(dataptr, q, 0, comm, module)

/**
 * @brief Pop from subqueue 0. Only contiguous types.
 */
#define sharm_queue_pop_contiguous(dataptr, q, comm, module) \
    _sharm_queue_pop_contiguous(dataptr, q, 0, comm, module)

/**
 * @brief Pop from subqueue sq.
 */
#define sharm_queue_pop_from_subqueue(dataptr, q, sq, comm, module) \
    _sharm_queue_pop(dataptr, q, sq, comm, module)

/**
 * @brief Pop from subqueue sq. Only contiguous types.
 */
#define sharm_queue_pop_from_subqueue_contiguous(dataptr, q, sq, comm, module) \
    _sharm_queue_pop_contiguous(dataptr, q, sq, comm, module)

/**
 * @brief Get control element value for queue.
 */
#define sharm_queue_get_ctrl(q, comm, module) \
    _sharm_queue_get_ctrl(q, 0, comm, module)

/**
 * @brief Get control element value for queue and subqueue.
 */
#define sharm_queue_get_subqueue_ctrl(q, sq, comm, module) \
    _sharm_queue_get_ctrl(q, sq, comm, module)

/**
 * @brief Clear control element value for queue.
 */
#define sharm_queue_clear_ctrl(q, comm, module) \
    _sharm_queue_clear_ctrl(q, 0, comm, module)

/**
 * @brief Get queue pointer.
 */
#define sharm_queue_get_ptr(q, comm, module) \
    _sharm_queue_get_ptr(q, 0, comm, module)

/*
 * Queue.
 */

size_t _sharm_queue_push(opal_convertor_t *convertor, size_t blocksize,
                         int queue, int subqueue, int notify,
                         ompi_communicator_t *comm,
                         mca_coll_sharm_module_t *module);
size_t _sharm_queue_pop(opal_convertor_t *convertor, int queue, int subqueue,
                        ompi_communicator_t *comm,
                        mca_coll_sharm_module_t *module);

size_t _sharm_queue_push_contiguous(const void *dataptr, size_t blocksize,
                                    int queue, int subqueue, int notify,
                                    ompi_communicator_t *comm,
                                    mca_coll_sharm_module_t *module);
size_t _sharm_queue_pop_contiguous(void *dataptr, int queue, int subqueue,
                                   ompi_communicator_t *comm,
                                   mca_coll_sharm_module_t *module);
long int _sharm_queue_get_ctrl(int queue, int subqueue,
                               ompi_communicator_t *comm,
                               mca_coll_sharm_module_t *module);
void *_sharm_queue_get_ptr(int queue, int subqueue, ompi_communicator_t *comm,
                           mca_coll_sharm_module_t *module);
void _sharm_queue_clear_ctrl(int queue, int subqueue, ompi_communicator_t *comm,
                             mca_coll_sharm_module_t *module);
// void adjust_queue_current_slot(int queue, int subqueue, int value,
//                                mca_coll_sharm_module_t *module);
int check_controls_is_zero(int queue, int subqueue, int slot,
                           ompi_communicator_t *comm,
                           mca_coll_sharm_module_t *module);

#endif /* MCA_COLL_SHARM_QUEUE_H */
