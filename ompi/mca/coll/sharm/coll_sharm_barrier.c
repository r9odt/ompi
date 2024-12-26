/** @file */

#include "coll_sharm.h"

/*
 * External variables.
 */

extern int mca_coll_sharm_stream;
extern int mca_coll_sharm_barrier_algorithm;

/*
 * Local functions.
 */

/**
 * @brief Sharm Barrier collective operation entrypoint.
 * @param[in] comm communicator for collective operation.
 * @param[in] module sharm module structure.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_barrier_intra(ompi_communicator_t *comm,
                        mca_coll_base_module_t *module)
{
    SHARM_INIT();
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;

    int ret = OMPI_SUCCESS;

    SHARM_NEW_OP(sharm_module);
    SHARM_NEW_COLL(barrier, sharm_module);

    if (ompi_comm_size(comm) < 2) {
        return MPI_SUCCESS;
    }

    OPAL_OUTPUT_VERBOSE(
        (SHARM_LOG_FUNCTION_CALL, mca_coll_sharm_stream,
         "coll:sharm:%d:barrier: (%d/%d/%s)", SHARM_COLL(barrier, sharm_module),
         ompi_comm_rank(comm), ompi_comm_size(comm), comm->c_name));

    if (!sharm_is_single_node_mode(comm)) {
        opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                            "coll:sharm:%d:barrier: (%d/%d/%s) "
                            "Operation cannot support multiple nodes, fallback",
                            SHARM_COLL(barrier, sharm_module),
                            ompi_comm_rank(comm), ompi_comm_size(comm),
                            comm->c_name);
        return sharm_module->fallbacks
            .fallback_barrier(comm,
                              sharm_module->fallbacks.fallback_barrier_module);
    }

    switch (mca_coll_sharm_barrier_algorithm) {
    case COLL_SHARM_BARRIER_ALG_SENSE_REVERSING:
        SHARM_PROFILING_TOTAL_TIME_START(sharm_module, barrier);
        ret = sharm_barrier_sense_reversing(comm, module);
        SHARM_PROFILING_TOTAL_TIME_STOP(sharm_module, barrier);
        return ret;
    case COLL_SHARM_BARRIER_ALG_CICO:
    default:
        break;
    }

    SHARM_PROFILING_TOTAL_TIME_START(sharm_module, barrier);
    ret = sharm_barrier_cico(comm, module);
    SHARM_PROFILING_TOTAL_TIME_STOP(sharm_module, barrier);
    return ret;
}

/**
 * @brief centralized sense-reversing Barrier algorithm.
 * Space: O(1)
 * Time: O(p)
 * @return OMPI_SUCCESS or error code.
 */
int sharm_barrier_sense_reversing(ompi_communicator_t *comm,
                                  mca_coll_base_module_t *module)
{
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;
    int comm_size = ompi_comm_size(comm);
    int err = MPI_SUCCESS;

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_CALL, mca_coll_sharm_stream,
                         "coll:sharm:%d:barrier_sr: (%d/%d/%s)",
                         SHARM_COLL(barrier, sharm_module),
                         ompi_comm_rank(comm), ompi_comm_size(comm),
                         comm->c_name));

    /*
     * Initial state:
     *   shm_sr_barrier_sense = comm_size
     *   shm_sr_barrier_sense = 1
     *   barrier_sr_pvt_sense = 1 (per process)
     */
    shm_data->barrier_sr_pvt_sense ^= 1;
    uint32_t pvt_sense = shm_data->barrier_sr_pvt_sense;
    opal_atomic_uint32_t *shm_barrier_sr_counter = shm_data
                                                       ->shm_barrier_sr_counter;
    uint32_t *shm_barrier_sr_sense = shm_data->shm_barrier_sr_sense;

    if (1
        == opal_atomic_fetch_add_32((opal_atomic_int32_t *)
                                        shm_barrier_sr_counter,
                                    -1)) {
        /* Last process releases all */
        *shm_barrier_sr_counter = comm_size;
        *shm_barrier_sr_sense = pvt_sense;
        opal_atomic_wmb();
    } else {
        SHARM_SPIN_CONDITION(*shm_barrier_sr_sense == pvt_sense);
    }

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_CALL, mca_coll_sharm_stream,
                         "coll:sharm:%d:barrier_sr: (%d/%d/%s) complete",
                         SHARM_COLL(barrier, sharm_module),
                         ompi_comm_rank(comm), ompi_comm_size(comm),
                         comm->c_name));
    return err;
}

/**
 * @brief shared-memory based algorithm for Barrier using CICO approach.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_barrier_cico(ompi_communicator_t *comm,
                       mca_coll_base_module_t *module)
{
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;

    int comm_size = ompi_comm_size(comm);
    int comm_rank = ompi_comm_rank(comm);
    int err = MPI_SUCCESS;

    // int barrier_root = SHARM_COLL(barrier, sharm_module) % comm_size;
    int barrier_root = 0;
    char barrier_data = SHARM_COLL(barrier, sharm_module);

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_CALL, mca_coll_sharm_stream,
                         "coll:sharm:%d:barrier_cico: (%d/%d/%s)",
                         SHARM_COLL(barrier, sharm_module),
                         ompi_comm_rank(comm), ompi_comm_size(comm),
                         comm->c_name));

    int qdatacnt = 0;
    if (comm_rank == barrier_root) {
        for (int i = 0; i < comm_size; ++i) {
            if (i == barrier_root) {
                continue;
            }
            wait_queue_func(qdatacnt,
                            sharm_queue_get_ctrl(i, comm, sharm_module));
            sharm_queue_clear_ctrl(i, comm, sharm_module);
        }
        wait_queue_func(qdatacnt,
                        sharm_queue_push_contiguous(&barrier_data, 1, comm_rank,
                                                    -1, comm, sharm_module));
    } else {
        wait_queue_func(qdatacnt,
                        sharm_queue_push_contiguous(&barrier_data, 1, comm_rank,
                                                    barrier_root, comm,
                                                    sharm_module));

        // Adjust slots counters for sync it.
        for (int i = 0; i < comm_size; ++i) {
            if (i == comm_rank || i == barrier_root) {
                continue;
            }
            adjust_queue_current_slot(i, 0, 1, sharm_module);
        }
        wait_queue_func(qdatacnt,
                        sharm_queue_get_ctrl(barrier_root, comm, sharm_module));
        sharm_queue_clear_ctrl(barrier_root, comm, sharm_module);
    }

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_CALL, mca_coll_sharm_stream,
                         "coll:sharm:%d:barrier_cico: (%d/%d/%s) complete",
                         SHARM_COLL(barrier, sharm_module),
                         ompi_comm_rank(comm), ompi_comm_size(comm),
                         comm->c_name));
    return err;
}

/**
 * @brief shared-memory based algorithm for gather barrier stage.
 * It is internal implementation for pause process when using ZeroCopy in
 * collectivies with root process.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_barrier_gather_cico(int root, ompi_communicator_t *comm,
                              mca_coll_base_module_t *module)
{
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;

    int comm_size = ompi_comm_size(comm);
    int comm_rank = ompi_comm_rank(comm);
    int err = MPI_SUCCESS;

    char barrier_data;

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_CALL, mca_coll_sharm_stream,
                         "coll:sharm:%d:barrier_gather: (%d/%d/%s)",
                         SHARM_OP(sharm_module), ompi_comm_rank(comm),
                         ompi_comm_size(comm), comm->c_name));

    if (comm_rank == root) {
        for (int i = 0; i < comm_size; ++i) {
            if (i == root) {
                continue;
            }
            wait_queue_func_no_return(
                sharm_queue_get_ctrl(i, comm, sharm_module));
            sharm_queue_clear_ctrl(i, comm, sharm_module);
        }
    } else {
        wait_queue_func_no_return(
            sharm_queue_push_contiguous(&barrier_data,
                                        sharm_module->shared_memory_data
                                            ->mu_cacheline_size,
                                        comm_rank, root, comm, sharm_module));

        // Adjust slots counters for sync it.
        for (int i = 0; i < comm_size; ++i) {
            if (i == comm_rank || i == root) {
                continue;
            }
            // (i ^ comm_rank) | (i ^ root);
            adjust_queue_current_slot(i, 0, 1, sharm_module);
        }
    }

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_CALL, mca_coll_sharm_stream,
                         "coll:sharm:%d:barrier_gather: (%d/%d/%s) complete",
                         SHARM_OP(sharm_module), ompi_comm_rank(comm),
                         ompi_comm_size(comm), comm->c_name));
    return err;
}

/**
 * @brief shared-memory based algorithm for scatter barrier stage.
 * It is internal implementation for pause process when using ZeroCopy in
 * collectivies with root process.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_barrier_bcast_cico(int root, ompi_communicator_t *comm,
                             mca_coll_base_module_t *module)
{
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;

    int comm_size = ompi_comm_size(comm);
    int comm_rank = ompi_comm_rank(comm);
    int err = MPI_SUCCESS;

    char barrier_data = SHARM_OP(sharm_module);

    if (comm_size < 2) {
        return err;
    }

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_CALL, mca_coll_sharm_stream,
                         "coll:sharm:%d:barrier_bcast: (%d/%d/%s)",
                         SHARM_OP(sharm_module), ompi_comm_rank(comm),
                         ompi_comm_size(comm), comm->c_name));

    int qdatacnt = 0;
    if (comm_rank == root) {
        wait_queue_func(qdatacnt,
                        sharm_queue_push_contiguous(
                            &barrier_data,
                            sharm_module->shared_memory_data->mu_cacheline_size,
                            comm_rank, -1, comm, sharm_module));
    } else {
        wait_queue_func(qdatacnt,
                        sharm_queue_get_ctrl(root, comm, sharm_module));
        sharm_queue_clear_ctrl(root, comm, sharm_module);
    }

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_CALL, mca_coll_sharm_stream,
                         "coll:sharm:%d:barrier_bcast: (%d/%d/%s) complete",
                         SHARM_OP(sharm_module), ompi_comm_rank(comm),
                         ompi_comm_size(comm), comm->c_name));
    return err;
}
