/** @file */

#include "coll_sharm.h"

/*
 * External variables.
 */

extern int mca_coll_sharm_stream;
extern int mca_coll_sharm_priority;
extern int mca_coll_sharm_verbose;

/*
 * External functions.
 */

/*
 * Local functions.
 */

static int sharm_module_enable(mca_coll_base_module_t *module,
                               ompi_communicator_t *comm);
int mca_coll_sharm_init_query(bool enable_progress_threads,
                              bool enable_mpi_threads);
mca_coll_base_module_t *mca_coll_sharm_comm_query(ompi_communicator_t *comm,
                                                  int *priority);
int sharm_cleanup_coll(mca_coll_sharm_module_t *module);

/**
 * @brief Initial query function that is invoked during MPI_INIT, allowing
 * this component to disqualify itself if it doesn't support the
 * required level of thread support.
 */
int mca_coll_sharm_init_query(bool enable_progress_threads,
                              bool enable_mpi_threads)
{
    if (enable_mpi_threads) {
        opal_output_verbose(
            SHARM_LOG_INFO, mca_coll_sharm_stream,
            "coll:sharm:module_init_query: sharm does not support "
            "MPI_THREAD_MULTIPLE");
        return OMPI_ERROR;
    }
    return OMPI_SUCCESS;
}

/**
 * @brief Invoked when there's a new communicator that has been created.
 * Look at the communicator and decide which set of functions and
 * priority we want to return.
 * @param[in] comm mpi communicator.
 * @param[in] priority component priority.
 * @return module instance.
 */
mca_coll_base_module_t *mca_coll_sharm_comm_query(ompi_communicator_t *comm,
                                                  int *priority)
{
    mca_coll_sharm_module_t *sharm_module;

    /*
     * SHARM module is created for each communicator,
     * but shared-memory collectives operations are initialized
     * lazly at first call on it communicator.
     */

    *priority = mca_coll_sharm_priority;
    if (mca_coll_sharm_priority <= 0) {
        opal_output_verbose(
            SHARM_LOG_INFO, mca_coll_sharm_stream,
            "coll:sharm:module_comm_query: (%d/%d/%s) too low priority %d",
            ompi_comm_rank(comm), ompi_comm_size(comm), comm->c_name,
            mca_coll_sharm_priority);
        return NULL;
    }

    if (ompi_comm_size(comm) < 2 || !sharm_is_single_node_mode(comm)) {
        opal_output_verbose(
            SHARM_LOG_INFO, mca_coll_sharm_stream,
            "coll:sharm:module_comm_query: invalid communicator %s, "
            "is too small (%d), "
            "or not all processes share a same computer node (%d)",
            comm->c_name, ompi_comm_size(comm),
            !sharm_is_single_node_mode(comm));
        return NULL;
    }

    if (OMPI_COMM_IS_INTER(comm)) {
        opal_output_verbose(
            SHARM_LOG_INFO, mca_coll_sharm_stream,
            "coll:sharm:module_comm_query: sharm does not support "
            "inter-communicators");
        *priority = 0;
        return NULL;
    }

    if (opal_using_threads()) {
        opal_output_verbose(
            SHARM_LOG_INFO, mca_coll_sharm_stream,
            "coll:sharm:module_comm_query: sharm does not support "
            "multi-threading collective calls");
        *priority = 0;
        return NULL;
    }

#if SHARM_CHECK_NUMA_SUPPORT
    opal_output_verbose(
        SHARM_LOG_INFO, mca_coll_sharm_stream,
        "coll:sharm:module_comm_query: component has NUMA support");
#else
    opal_output_verbose(
        SHARM_LOG_INFO, mca_coll_sharm_stream,
        "coll:sharm:module_comm_query: component does not have NUMA support");
#endif

#if SHARM_CHECK_KNEM_SUPPORT
    opal_output_verbose(
        SHARM_LOG_INFO, mca_coll_sharm_stream,
        "coll:sharm:module_comm_query: component has KNEM support");
#else
    opal_output_verbose(
        SHARM_LOG_INFO, mca_coll_sharm_stream,
        "coll:sharm:module_comm_query: component does not have KNEM support");
#endif

#if SHARM_CHECK_XPMEM_SUPPORT
    opal_output_verbose(
        SHARM_LOG_INFO, mca_coll_sharm_stream,
        "coll:sharm:module_comm_query: component has XPMEM support");
#else
    opal_output_verbose(
        SHARM_LOG_INFO, mca_coll_sharm_stream,
        "coll:sharm:module_comm_query: component does not have XPMEM support");
#endif

#if SHARM_CHECK_CMA_SUPPORT
    opal_output_verbose(
        SHARM_LOG_INFO, mca_coll_sharm_stream,
        "coll:sharm:module_comm_query: component has CMA support");
#else
    opal_output_verbose(
        SHARM_LOG_INFO, mca_coll_sharm_stream,
        "coll:sharm:module_comm_query: component does not have CMA support");
#endif

#if SHARM_CHECK_AVX2_SUPPORT
    opal_output_verbose(
        SHARM_LOG_INFO, mca_coll_sharm_stream,
        "coll:sharm:module_comm_query: component has AVX2 support");
#else
    opal_output_verbose(
        SHARM_LOG_INFO, mca_coll_sharm_stream,
        "coll:sharm:module_comm_query: component does not have AVX2 support");
#endif

    sharm_module = OBJ_NEW(mca_coll_sharm_module_t);
    if (NULL == sharm_module) {
        return NULL;
    }

    sharm_module->super.coll_module_enable = sharm_module_enable;
    sharm_module->super.coll_allgather = sharm_allgather_intra;
    sharm_module->super.coll_allgatherv = sharm_allgatherv_intra;
    sharm_module->super.coll_allreduce = sharm_allreduce_intra;
    sharm_module->super.coll_alltoall = sharm_alltoall_intra;
    sharm_module->super.coll_alltoallv = sharm_alltoallv_intra;
    sharm_module->super.coll_alltoallw = sharm_alltoallw_intra;
    sharm_module->super.coll_barrier = sharm_barrier_intra;
    sharm_module->super.coll_bcast = sharm_bcast_intra;
    sharm_module->super.coll_exscan = sharm_exscan_intra;
    sharm_module->super.coll_gather = sharm_gather_intra;
    sharm_module->super.coll_gatherv = sharm_gatherv_intra;
    sharm_module->super.coll_reduce = sharm_reduce_intra;
    sharm_module->super.coll_reduce_scatter_block
        = sharm_reduce_scatter_block_intra;
    sharm_module->super.coll_reduce_scatter = sharm_reduce_scatter_intra;
    sharm_module->super.coll_scan = sharm_scan_intra;
    sharm_module->super.coll_scatter = sharm_scatter_intra;
    sharm_module->super.coll_scatterv = sharm_scatterv_intra;

    sharm_module->comm = comm;
    sharm_module->shared_memory_data = NULL;

    opal_output_verbose(
        SHARM_LOG_INFO, mca_coll_sharm_stream,
        "coll:sharm:module_comm_query: (%d/%d/%s) sharm component initialized",
        ompi_comm_rank(comm), ompi_comm_size(comm), comm->c_name);
    return &(sharm_module->super);
}

/**
 * @brief Init module on the communicator.
 * @param[in] module sharm module structure.
 * @param[in] comm mpi communicator.
 * @return OMPI_SUCCESS or error code.
 */
static int sharm_module_enable(mca_coll_base_module_t *module,
                               ompi_communicator_t *comm)
{
    int err = OMPI_SUCCESS;
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_module->comm = comm;
    mca_coll_base_comm_t *data = NULL;

    /* Save previous component's fallback information */

    sharm_module->fallback_barrier = comm->c_coll->coll_barrier;
    sharm_module->fallback_barrier_module = comm->c_coll->coll_barrier_module;
    OBJ_RETAIN(sharm_module->fallback_barrier_module);

    sharm_module->fallback_reduce = comm->c_coll->coll_reduce;
    sharm_module->fallback_reduce_module = comm->c_coll->coll_reduce_module;
    OBJ_RETAIN(sharm_module->fallback_reduce_module);

    sharm_module->fallback_allreduce = comm->c_coll->coll_allreduce;
    sharm_module->fallback_allreduce_module = comm->c_coll
                                                  ->coll_allreduce_module;
    OBJ_RETAIN(sharm_module->fallback_allreduce_module);

    sharm_module->fallback_scan = comm->c_coll->coll_scan;
    sharm_module->fallback_scan_module = comm->c_coll->coll_scan_module;
    OBJ_RETAIN(sharm_module->fallback_scan_module);

    sharm_module->fallback_exscan = comm->c_coll->coll_exscan;
    sharm_module->fallback_exscan_module = comm->c_coll->coll_exscan_module;
    OBJ_RETAIN(sharm_module->fallback_exscan_module);

    /*
     * Prepare the placeholder for the array of request.
     */
    data = OBJ_NEW(mca_coll_base_comm_t);
    if (NULL == data) {
        return OMPI_ERROR;
    }
    sharm_module->super.base_data = data;

    if (MPI_COMM_WORLD == comm) {
        /*
         * Initialize shared-memory segment.
         */
        err = mca_coll_sharm_init_segment(module);
        OPAL_OUTPUT_VERBOSE(
            (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
             "coll:sharm: (%d/%d/%s), my mca_coll_sharm_init_segment is %d",
             ompi_comm_rank(comm), ompi_comm_size(comm), comm->c_name, err));
        if (OMPI_SUCCESS != err) {
            return err;
        }
    }

    return err;
}

/**
 * @brief free shared-memory segment.
 */
int sharm_cleanup_coll(mca_coll_sharm_module_t *module)
{
    int err = OMPI_SUCCESS;

    /*
     * module->shared_memory_data
     */
    if (NULL != module->shared_memory_data) {
#if SHARM_CHECK_XPMEM_SUPPORT
        xpmem_segid_t my_seg_id = *(
            (xpmem_segid_t *) module->shared_memory_data
                ->xpmem_segid[ompi_comm_rank(module->comm)]);
        if (-1 != my_seg_id) {
            xpmem_remove(my_seg_id);
        }
        free(module->shared_memory_data->xpmem_segid);
#endif

        if (NULL != module->shared_memory_data->current_queue_slots) {
            free(module->shared_memory_data->current_queue_slots);
            module->shared_memory_data->current_queue_slots = NULL;
        }

        if (NULL != module->shared_memory_data->shm_queue) {
            free(module->shared_memory_data->shm_queue);
            module->shared_memory_data->shm_queue = NULL;
        }

        if (NULL != module->shared_memory_data->shm_control) {
            free(module->shared_memory_data->shm_control);
            module->shared_memory_data->shm_control = NULL;
        }

        if (NULL != module->shared_memory_data->shm_process_ids) {
            free(module->shared_memory_data->shm_process_ids);
            module->shared_memory_data->shm_process_ids = NULL;
        }

        if (NULL != module->shared_memory_data) {
            free(module->shared_memory_data);
            module->shared_memory_data = NULL;
        }
    }

    if (NULL != module->local_op_memory_map) {
        free(module->local_op_memory_map);
        module->local_op_memory_map = NULL;
    }

    /*
     * module->local_collectivies_info.memory
     */

    if (NULL != module->local_collectivies_info.pointers_memory) {
        free(module->local_collectivies_info.pointers_memory);
        module->local_collectivies_info.pointers_memory = NULL;
    }

    if (NULL != module->local_collectivies_info.memory) {
        free(module->local_collectivies_info.memory);
        module->local_collectivies_info.memory = NULL;
    }

    // TODO: Free shared memory segment
    // sharm_coll_data_t *c = module->shared_memory_data;

    // if (NULL != c) {
    //     /* Munmap the per-communicator shmem data segment */
    //     if (NULL != c->segmeta) {
    //         /* Ignore any errors -- what are we going to do about
    //            them? */
    //         mca_common_sm_fini(c->segmeta);
    //         OBJ_RELEASE(c->segmeta);
    //     }
    //     free(c);
    // }

    return err;
}

/**
 * @brief mca_coll_sharm_module_t_class instance construct.
 */
static void mca_coll_sharm_module_construct(mca_coll_sharm_module_t *module)
{
    module->comm = MPI_COMM_NULL;
    module->shared_memory_data = NULL;
}

/**
 * @brief mca_coll_sharm_module_t_class instance destruct.
 */
static void mca_coll_sharm_module_destruct(mca_coll_sharm_module_t *module)
{
    /*
     * Call barrier for cma, xpmem, knem support.
     * Use fallback module or pml to avoid errors if segment was not initialized
     */
    // module->fallback_barrier(module->comm,module->fallback_barrier_module);
    // char barrier_data = 0;
    // if (ompi_comm_rank(module->comm) == 0) {
    //     for (int i = 1; i < ompi_comm_size(module->comm); i++) {
    //         MCA_PML_CALL(recv(&barrier_data, sizeof(barrier_data), MPI_BYTE,
    //         i,
    //                           MCA_COLL_BASE_TAG_BARRIER, module->comm,
    //                           MPI_STATUS_IGNORE));
    //     }
    //     for (int i = 1; i < ompi_comm_size(module->comm); i++) {
    //         MCA_PML_CALL(send(&barrier_data, sizeof(barrier_data), MPI_BYTE,
    //         i,
    //                           MCA_COLL_BASE_TAG_BARRIER,
    //                           MCA_PML_BASE_SEND_STANDARD, module->comm));
    //     }
    // } else {
    //     MCA_PML_CALL(send(&barrier_data, sizeof(barrier_data), MPI_BYTE, 0,
    //                       MCA_COLL_BASE_TAG_BARRIER,
    //                       MCA_PML_BASE_SEND_STANDARD, module->comm));
    //     MCA_PML_CALL(recv(&barrier_data, sizeof(barrier_data), MPI_BYTE, 0,
    //                       MCA_COLL_BASE_TAG_BARRIER, module->comm,
    //                       MPI_STATUS_IGNORE));
    // }

    if (NULL != module->shared_memory_data) {
        SHARM_PROFILING_DUMP_ALL_GLOBAL_VALUES(module);
    }

#if SHARM_CHECK_XPMEM_SUPPORT
    if (NULL != module->apids) {
        for (int i = 0; i < ompi_comm_rank(module->comm); ++i) {
            xpmem_apid_t apid = module->apids[i];
            if (-1 != apid) {
                xpmem_release(apid);
            }
        }
        free(module->apids);
    }
#endif

    sharm_cleanup_coll(module);

    if (NULL != module->fallback_barrier_module) {
        OBJ_RELEASE(module->fallback_barrier_module);
    }

    if (NULL != module->fallback_reduce_module) {
        OBJ_RELEASE(module->fallback_reduce_module);
    }
    if (NULL != module->fallback_allreduce_module) {
        OBJ_RELEASE(module->fallback_allreduce_module);
    }
    if (NULL != module->fallback_scan_module) {
        OBJ_RELEASE(module->fallback_scan_module);
    }
    if (NULL != module->fallback_exscan_module) {
        OBJ_RELEASE(module->fallback_exscan_module);
    }
}

/**
 * @brief mca_coll_sharm_module_t_class instance declaration
 */
OBJ_CLASS_INSTANCE(mca_coll_sharm_module_t, mca_coll_base_module_t,
                   mca_coll_sharm_module_construct,
                   mca_coll_sharm_module_destruct);
