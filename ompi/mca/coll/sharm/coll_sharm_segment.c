/** @file */

#include "coll_sharm.h"

/*
 * External variables.
 */

extern int mca_coll_sharm_stream;
extern int mca_coll_sharm_cacheline_size;
extern int mca_coll_sharm_nfrags;
extern int mca_coll_sharm_fragment_size;
extern char *mca_coll_sharm_segment_path;

/**
 * @brief allocates shared-memory segments.
 * @param[in] module sharm module structure.
 * @return OMPI_SUCCESS or error code.
 */
int mca_coll_sharm_init_segment(mca_coll_base_module_t *module)
{
    int err = OMPI_SUCCESS;
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    ompi_communicator_t *comm = sharm_module->comm;
    int comm_size = ompi_comm_size(comm);
    int node_comm_size = ompi_group_count_local_peers(comm->c_local_group);

    unsigned char *base = NULL;
    /* If no session directory was created, then we cannot be used */
    if (NULL == ompi_process_info.job_session_dir) {
        err = OMPI_ERR_OUT_OF_RESOURCE;
        return err;
    }

    if (NULL != sharm_module->shared_memory_data || comm_size < 2
        || !sharm_is_single_node_mode(comm)) {
        return err;
    }

    int comm_rank = ompi_comm_rank(comm);

    sharm_module->shared_memory_data = calloc(1, sizeof(sharm_coll_data_t));
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;
    if (NULL == shm_data) {
        err = OMPI_ERR_OUT_OF_RESOURCE;
        return err;
    }

    shm_data->mu_cacheline_size = mca_coll_sharm_cacheline_size;
    shm_data->mu_queue_nfrags = mca_coll_sharm_nfrags;

    /* Allocate shared-memory region */
    err = sharm_allocate_segment(module);
    if (OMPI_SUCCESS != err) {
        err = OMPI_ERR_OUT_OF_RESOURCE;
        return err;
    }

    /*
     * Setup local fields.
     */
    sharm_module->local_barrier = 0;
    sharm_module->local_bcast = 0;
    sharm_module->local_gather = 0;
    sharm_module->local_gatherv = 0;
    sharm_module->local_scatter = 0;
    sharm_module->local_scatterv = 0;
    sharm_module->local_alltoall = 0;
    sharm_module->local_alltoallv = 0;
    sharm_module->local_alltoallw = 0;
    sharm_module->local_allgather = 0;
    sharm_module->local_allgatherv = 0;
    sharm_module->local_scan = 0;
    sharm_module->local_exscan = 0;
    sharm_module->local_reduce = 0;
    sharm_module->local_reduce_scatter = 0;
    sharm_module->local_reduce_scatter_block = 0;
    sharm_module->local_allreduce = 0;
    sharm_module->local_op = 0;

    /*
     * Init profiling fields.
     */
    SHARM_INIT_GLOBAL_PROFILING_COUNTERS(sharm_module, barrier);
    SHARM_INIT_GLOBAL_PROFILING_COUNTERS(sharm_module, bcast);
    SHARM_INIT_GLOBAL_PROFILING_COUNTERS(sharm_module, gather);
    SHARM_INIT_GLOBAL_PROFILING_COUNTERS(sharm_module, gatherv);
    SHARM_INIT_GLOBAL_PROFILING_COUNTERS(sharm_module, scatter);
    SHARM_INIT_GLOBAL_PROFILING_COUNTERS(sharm_module, scatterv);
    SHARM_INIT_GLOBAL_PROFILING_COUNTERS(sharm_module, alltoall);
    SHARM_INIT_GLOBAL_PROFILING_COUNTERS(sharm_module, alltoallv);
    SHARM_INIT_GLOBAL_PROFILING_COUNTERS(sharm_module, alltoallw);
    SHARM_INIT_GLOBAL_PROFILING_COUNTERS(sharm_module, allgather);
    SHARM_INIT_GLOBAL_PROFILING_COUNTERS(sharm_module, allgatherv);
    SHARM_INIT_GLOBAL_PROFILING_COUNTERS(sharm_module, scan);
    SHARM_INIT_GLOBAL_PROFILING_COUNTERS(sharm_module, exscan);
    SHARM_INIT_GLOBAL_PROFILING_COUNTERS(sharm_module, reduce);
    SHARM_INIT_GLOBAL_PROFILING_COUNTERS(sharm_module, reduce_scatter);
    SHARM_INIT_GLOBAL_PROFILING_COUNTERS(sharm_module, reduce_scatter_block);
    SHARM_INIT_GLOBAL_PROFILING_COUNTERS(sharm_module, allreduce);

    /*
     * Setup memory map for support operations
     * Alltoall fields (Also used by other operations)
     *
     * rtotal_sizes_by_rank - sizeof(size_t) * comm_size
     * stotal_sizes_by_rank - sizeof(size_t) * comm_size
     * recv_bytes_by_rank - sizeof(size_t) * comm_size
     * rconvertors_by_rank - sizeof(opal_convertor_t) * comm_size
     * send_bytes_by_rank - sizeof(size_t) * comm_size
     * sconvertors_by_rank - sizeof(opal_convertor_t) * comm_size
     * recv_buff_ptr_for_rank - sizeof(char *) * comm_size
     * send_buff_ptr_for_rank - sizeof(char *) * comm_size
     *
     * Reduce fields:
     * 2 * fragment size
     * nchilds - currently assume that allocated memory is enouth for this
     *
     * Collectivies info exchange v1 (Deprecated):
     * rcounts_convertors - sizeof(opal_convertor_t) * comm_size
     * counts_bytes_received_by_rank - sizeof(size_t) * comm_size
     * stotal_sizes_by_rank_for_all_ranks - sizeof(size_t) * comm_size *
     * comm_size
     */

    sharm_module->local_op_memory_map = malloc(
        //
        comm_size
            * (2 * sizeof(opal_convertor_t) + 4 * sizeof(size_t)
               + 2 * sizeof(char *))
        + 2 * shm_data->mu_queue_fragment_size
        // Collectivies info exchange v1 (Deprecated)
        // + (sizeof(size_t) + sizeof(opal_convertor_t)) * comm_size
        // + comm_size * comm_size * sizeof(size_t)
    );

    if (NULL == sharm_module->local_op_memory_map) {
        err = OMPI_ERR_OUT_OF_RESOURCE;
        return err;
    }

    /*
     * Collectivies info exchange v2:
     * For each rank created:
     * scounts - sizeof(size_t) * comm_size
     * rcounts - sizeof(size_t) * comm_size
     * sbuf - sizeof(ptrdiff_t) * comm_size <- for each rank (alltoall)
     * rbuf - sizeof(ptrdiff_t) * comm_size <- for each rank (alltoall)
     * sdtypes_ext - sizeof(ptrdiff_t) * comm_size
     * sdtypes_size - sizeof(size_t) * comm_size
     * sdtypes_contiguous - sizeof(char) * comm_size
     * rdtypes_ext - sizeof(ptrdiff_t) * comm_size
     * rdtypes_size - sizeof(size_t) * comm_size
     * rdtypes_contiguous - sizeof(char) * comm_size
     *
     * Collectivies info exchange v2.1:
     * Free placement of blocks in memory depending on the algorithm
     */

    sharm_local_collectivies_data_t *coll_info = &(
        sharm_module->local_collectivies_info);
    coll_info->one_rank_block_size = comm_size
                                         * (4 * sizeof(ptrdiff_t)
                                            + 4 * sizeof(size_t))
                                     + 2 * sizeof(char);

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_DEBUG, mca_coll_sharm_stream,
                         "coll:sharm: comm (%d/%d/%s): "
                         "coll_info one_rank_block_size is %d",
                         comm_rank, comm_size, comm->c_name,
                         coll_info->one_rank_block_size));

    coll_info->memory = calloc(comm_size, coll_info->one_rank_block_size);
    coll_info->pointers_memory = calloc(comm_size, sizeof(ptrdiff_t *) * 4
                                                       + sizeof(size_t *) * 4
                                                       + sizeof(char *) * 2
#if SHARM_CHECK_XPMEM_SUPPORT
                                                       + sizeof(xpmem_segid_t *)
#endif
    );

    if (NULL == coll_info->memory || NULL == coll_info->pointers_memory) {
        err = OMPI_ERR_OUT_OF_RESOURCE;
        return err;
    }

    base = (unsigned char *) coll_info->pointers_memory;
    coll_info->sbuf = (ptrdiff_t **) base;
    base += comm_size * sizeof(ptrdiff_t *);
    coll_info->rbuf = (ptrdiff_t **) base;
    base += comm_size * sizeof(ptrdiff_t *);
    coll_info->sdtypes_ext = (ptrdiff_t **) base;
    base += comm_size * sizeof(ptrdiff_t *);
    coll_info->rdtypes_ext = (ptrdiff_t **) base;
    base += comm_size * sizeof(ptrdiff_t *);
    coll_info->scounts = (size_t **) base;
    base += comm_size * sizeof(size_t *);
    coll_info->rcounts = (size_t **) base;
    base += comm_size * sizeof(size_t *);
    coll_info->sdtypes_size = (size_t **) base;
    base += comm_size * sizeof(size_t *);
    coll_info->rdtypes_size = (size_t **) base;
    base += comm_size * sizeof(size_t *);
    coll_info->sdtypes_contiguous = (char **) base;
    base += comm_size * sizeof(char *);
    coll_info->rdtypes_contiguous = (char **) base;
    base += comm_size * sizeof(char *);

    base = (unsigned char *) coll_info->memory;

    for (int i = 0; i < comm_size; ++i) {
        base = (unsigned char *) RESOLVE_COLLECTIVIES_DATA(sharm_module, i);

        coll_info->sdtypes_contiguous[i] = (char *) base;
        base += comm_size * sizeof(char);
        coll_info->sbuf[i] = (ptrdiff_t *) base;
        base += comm_size * sizeof(ptrdiff_t);
        coll_info->rdtypes_contiguous[i] = (char *) base;
        base += comm_size * sizeof(char);
        coll_info->rbuf[i] = (ptrdiff_t *) base;
        base += comm_size * sizeof(ptrdiff_t);
        coll_info->sdtypes_ext[i] = (ptrdiff_t *) base;
        base += comm_size * sizeof(ptrdiff_t);
        coll_info->rdtypes_ext[i] = (ptrdiff_t *) base;
        base += comm_size * sizeof(ptrdiff_t);
        coll_info->scounts[i] = (size_t *) base;
        base += comm_size * sizeof(size_t);
        coll_info->rcounts[i] = (size_t *) base;
        base += comm_size * sizeof(size_t);
        coll_info->sdtypes_size[i] = (size_t *) base;
        base += comm_size * sizeof(size_t);
        coll_info->rdtypes_size[i] = (size_t *) base;
        base += comm_size * sizeof(size_t);
    }

    /*
     * Setup local barrier field.
     */
    shm_data->barrier_sr_pvt_sense = 1;

    /*
     * Setup local queue slots counters.
     */
    shm_data->nproc = comm_size;

    shm_data->current_queue_slots = (uint32_t *) calloc(shm_data->nproc
                                                            * shm_data->nproc,
                                                        sizeof(uint32_t));
    if (NULL == shm_data->current_queue_slots) {
        err = OMPI_ERR_OUT_OF_RESOURCE;
        return err;
    }

    /*
     * Setup pointers to shared fields.
     */

    base = shm_data->segmeta->module_data_addr;

    /* Sense-reversing barrier */

    shm_data->shm_barrier_sr_counter = (opal_atomic_uint32_t *) (base);
    shm_data->shm_barrier_sr_sense = (uint32_t *) (base
                                                   + shm_data
                                                         ->mu_cacheline_size);
    if (0 == comm_rank) {
        *shm_data->shm_barrier_sr_counter = comm_size;
        *shm_data->shm_barrier_sr_sense = 1;
    }
    base += shm_data->mu_barrier_block_size; // go next block

    // TODO: Allocate one block for all pointers

    /* Process IDs storage block */

    shm_data->shm_process_ids = (unsigned char **) malloc(
        sizeof(unsigned char *) * comm_size);
    for (int i = 0; i < comm_size; ++i) {
        shm_data->shm_process_ids[i] = base + shm_data->mu_cacheline_size * i;
    }

    *((pid_t *) shm_data->shm_process_ids[comm_rank]) = getpid();

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_INFO, mca_coll_sharm_stream,
                         "coll:sharm: (%d/%d/%s), my pid is %d", comm_rank,
                         comm_size, comm->c_name,
                         SHARM_GET_RANK_PID(shm_data, comm_rank)));

    base += shm_data->mu_pids_block_size; // go next block

#if SHARM_CHECK_XPMEM_SUPPORT
    sharm_module->xpmem_runtime_check_support = SHARM_TRUE;
    shm_data->xpmem_segid = (xpmem_segid_t **) malloc(sizeof(xpmem_segid_t *)
                                                      * comm_size);
    for (int i = 0; i < comm_size; ++i) {
        shm_data->xpmem_segid[i]
            = (xpmem_segid_t *) (base + shm_data->mu_cacheline_size * i);
    }

    sharm_module->apids = (xpmem_apid_t *) calloc(comm_size,
                                                  sizeof(xpmem_apid_t));
    if (NULL == sharm_module->apids) {
        err = OMPI_ERR_OUT_OF_RESOURCE;
        return err;
    }

    /*
     * TODO: Map the whole address space. Is it faster ?.
     */

    xpmem_segid_t my_seg_id = xpmem_make(0, XPMEM_MAXADDR_SIZE,
                                         XPMEM_PERMIT_MODE, (void *) 0666);

    /*
     * If xpmem unreacheble just disable it in runtime
     * instead crash all component.
     */

    if (-1 == my_seg_id) {
        sharm_module->xpmem_runtime_check_support = SHARM_FALSE;

        OPAL_OUTPUT_VERBOSE(
            (SHARM_LOG_WARNING, mca_coll_sharm_stream,
             "coll:sharm: (%d/%d/%s), XPMEM disabled in runtime", comm_rank,
             comm_size, comm->c_name));
        // err = OMPI_ERR_OUT_OF_RESOURCE;
        // return err;
    };

    *((xpmem_segid_t *) shm_data->xpmem_segid[comm_rank]) = my_seg_id;

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_INFO, mca_coll_sharm_stream,
                         "coll:sharm: (%d/%d/%s), my XPMEM segid is %d",
                         comm_rank, comm_size, comm->c_name, my_seg_id));

    // Perform barrier because we need to xpmem_make completed before xpmem_get
    // sharm_barrier_sense_reversing(sharm_module, comm);

    for (int i = 0; i < comm_size; ++i) {
        // if (i == comm_rank) {
        //     continue;
        // }
        sharm_module->apids[i] = -1;
    }

    base += shm_data->mu_xpmem_segid_block_size; // go next block
#endif

    /* Queue control system */
    shm_data->shm_control = (unsigned char **) malloc(sizeof(unsigned char *)
                                                      * comm_size);
    for (int i = 0; i < comm_size; ++i) {
        shm_data->shm_control[i] = base
                                   + shm_data->mu_control_one_queue_system_size
                                         * i;
    }
    base += shm_data->mu_control_block_size; // go next block

    /* Queue system */
    shm_data->shm_queue = (unsigned char **) malloc(sizeof(unsigned char *)
                                                    * comm_size);
    for (int i = 0; i < comm_size; ++i) {
        shm_data->shm_queue[i] = base
                                 + shm_data->mu_queue_one_queue_system_size * i;
    }
    base += shm_data->mu_queue_block_size; // go next block

#if SHARM_CHECK_NUMA_SUPPORT
    if (0 == comm_rank) {
        sharm_check_pages_mem_affinity(comm_rank,
                                       shm_data->shm_barrier_sr_counter,
                                       shm_data->mu_barrier_block_size,
                                       "barrier");
    }
    sharm_check_pages_mem_affinity(comm_rank, shm_data->shm_queue[comm_rank],
                                   shm_data->mu_queue_one_queue_system_size,
                                   "queue");
    sharm_check_pages_mem_affinity(comm_rank, shm_data->shm_control[comm_rank],
                                   shm_data->mu_control_one_queue_system_size,
                                   "control");
#endif

    /* Indicate that we have successfully attached and setup */

    opal_atomic_add(&(shm_data->segmeta->module_seg->seg_inited), 1);
    opal_output_verbose(
        SHARM_LOG_INFO, mca_coll_sharm_stream,
        "coll:sharm: proc %d/%s: waiting on comm for peers to attach",
        comm_rank, comm->c_name);
    SHARM_SPIN_CONDITION(comm_size
                         == shm_data->segmeta->module_seg->seg_inited);

    /* Once we're all here, remove the mmap file; it's not needed anymore */

    if (0 == comm_rank) {
        unlink(shm_data->segmeta->shmem_ds.seg_name);
        opal_output_verbose(SHARM_LOG_INFO, mca_coll_sharm_stream,
                            "coll:sharm: proc %s/%s: removed mmap file %s",
                            ompi_comm_print_cid(comm), comm->c_name,
                            shm_data->segmeta->shmem_ds.seg_name);
    }

    return err;
}

/**
 * @brief creates shared-memory segment.
 * @param[in] module sharm module structure.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_allocate_segment(mca_coll_base_module_t *module)
{
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;
    ompi_communicator_t *comm = sharm_module->comm;
    int comm_size = ompi_group_count_local_peers(comm->c_local_group);
    int comm_rank = ompi_comm_rank(comm);
    char *shortpath = NULL, *fullpath = NULL;

    /*
     * Make the rendezvous filename for this comm shared-memory segment
     * The CID is not guaranteed to be unique among all procs on this node,
     * so also pair it with the PID of the proc with the lowest ORTE name
     * to form a unique filename.
     */

    ompi_proc_t *proc = ompi_group_peer_lookup(comm->c_local_group, 0);
    ompi_process_name_t *lowest_name = OMPI_CAST_RTE_NAME(
        &proc->super.proc_name);
    for (int i = 1; i < comm_size; i++) {
        proc = ompi_group_peer_lookup(comm->c_local_group, i);
        if (ompi_rte_compare_name_fields(OMPI_RTE_CMP_ALL,
                                         OMPI_CAST_RTE_NAME(
                                             &proc->super.proc_name),
                                         lowest_name)
            < 0) {
            lowest_name = OMPI_CAST_RTE_NAME(&proc->super.proc_name);
        }
    }

    opal_asprintf(&shortpath, "coll-sharm-cid-%s-name-%s.mmap",
                  ompi_comm_print_cid(comm), OMPI_NAME_PRINT(lowest_name));
    if (NULL == shortpath) {
        opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                            "coll:sharm: comm (%s/%s): asprintf failed",
                            ompi_comm_print_cid(comm), comm->c_name);
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    opal_output_verbose(
        SHARM_LOG_INFO, mca_coll_sharm_stream,
        "coll:sharm: comm (%s/%s): mca_coll_sharm_segment_path is %s",
        ompi_comm_print_cid(comm), comm->c_name,
        mca_coll_sharm_segment_path ? mca_coll_sharm_segment_path : "NULL");
    fullpath = opal_os_path(false,
                            mca_coll_sharm_segment_path > 0
                                ? mca_coll_sharm_segment_path
                                : ompi_process_info.job_session_dir,
                            shortpath, NULL);
    free(shortpath);
    if (NULL == fullpath) {
        opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                            "coll:sharm: comm (%s/%s): opal_os_path failed",
                            ompi_comm_print_cid(comm), comm->c_name);
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    /*
     * Calculate shared-memory block size
     */

    int ps = opal_getpagesize();
    shm_data->mu_page_size = ps;

    /* Barrier */
    shm_data->mu_barrier_block_size
        = ps * sharm_get_npages(2, shm_data->mu_cacheline_size);
    OPAL_OUTPUT_VERBOSE((SHARM_LOG_DEBUG, mca_coll_sharm_stream,
                         "coll:sharm: comm (%d/%d/%s): "
                         "mu_barrier_block_size is %ld",
                         comm_rank, comm_size, comm->c_name,
                         shm_data->mu_barrier_block_size));

    /* Queue system */
    /*
     * for each rank:
     * [PS] * sharm_get_npages(mca_coll_sharm_fragment_size) * nfrags *
     * comm_size
     */

    shm_data->mu_queue_fragment_size
        = ps * sharm_get_npages(1, mca_coll_sharm_fragment_size);
    shm_data->mu_queue_one_queue_size = shm_data->mu_queue_nfrags
                                        * shm_data->mu_queue_fragment_size;
    shm_data->mu_queue_one_queue_system_size = comm_size
                                               * shm_data
                                                     ->mu_queue_one_queue_size;
    shm_data->mu_queue_block_size = comm_size
                                    * shm_data->mu_queue_one_queue_system_size;

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_DEBUG, mca_coll_sharm_stream,
                         "coll:sharm: comm (%d/%d/%s): "
                         "mu_queue_fragment_size is %ld",
                         comm_rank, comm_size, comm->c_name,
                         shm_data->mu_queue_fragment_size));
    OPAL_OUTPUT_VERBOSE((SHARM_LOG_DEBUG, mca_coll_sharm_stream,
                         "coll:sharm: comm (%d/%d/%s): "
                         "mu_queue_one_queue_size is %ld",
                         comm_rank, comm_size, comm->c_name,
                         shm_data->mu_queue_one_queue_size));
    OPAL_OUTPUT_VERBOSE((SHARM_LOG_DEBUG, mca_coll_sharm_stream,
                         "coll:sharm: comm (%d/%d/%s): "
                         "mu_queue_one_queue_system_size is %ld",
                         comm_rank, comm_size, comm->c_name,
                         shm_data->mu_queue_one_queue_system_size));
    OPAL_OUTPUT_VERBOSE((SHARM_LOG_DEBUG, mca_coll_sharm_stream,
                         "coll:sharm: comm (%d/%d/%s): "
                         "mu_queue_block_size is %ld",
                         comm_rank, comm_size, comm->c_name,
                         shm_data->mu_queue_block_size));

    /* Queue control system */

    /*
     * for each rank:
     * [PS] * sharm_get_npages(comm_size *
     * sharm_cacheline_size) * comm_size * nfrags
     */

    shm_data->mu_control_fragment_size
        = ps * sharm_get_npages(comm_size, shm_data->mu_cacheline_size);
    shm_data->mu_control_one_queue_size = shm_data->mu_queue_nfrags
                                          * shm_data->mu_control_fragment_size;
    shm_data->mu_control_one_queue_system_size
        = comm_size * shm_data->mu_control_one_queue_size;
    shm_data->mu_control_block_size = comm_size
                                      * shm_data
                                            ->mu_control_one_queue_system_size;

    /* Process IDs storage block */

    shm_data->mu_pids_block_size
        = ps * sharm_get_npages(comm_size, shm_data->mu_cacheline_size);

#if SHARM_CHECK_XPMEM_SUPPORT
    shm_data->mu_xpmem_segid_block_size
        = ps * sharm_get_npages(comm_size, shm_data->mu_cacheline_size);
#endif

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_DEBUG, mca_coll_sharm_stream,
                         "coll:sharm: comm (%d/%d/%s): "
                         "mu_control_fragment_size is %ld",
                         comm_rank, comm_size, comm->c_name,
                         shm_data->mu_control_fragment_size));
    OPAL_OUTPUT_VERBOSE((SHARM_LOG_DEBUG, mca_coll_sharm_stream,
                         "coll:sharm: comm (%d/%d/%s): "
                         "mu_control_one_queue_size is %ld",
                         comm_rank, comm_size, comm->c_name,
                         shm_data->mu_control_one_queue_size));
    OPAL_OUTPUT_VERBOSE((SHARM_LOG_DEBUG, mca_coll_sharm_stream,
                         "coll:sharm: comm (%d/%d/%s): "
                         "mu_control_one_queue_system_size is %ld",
                         comm_rank, comm_size, comm->c_name,
                         shm_data->mu_control_one_queue_system_size));
    OPAL_OUTPUT_VERBOSE((SHARM_LOG_DEBUG, mca_coll_sharm_stream,
                         "coll:sharm: comm (%d/%d/%s): "
                         "mu_control_block_size is %ld",
                         comm_rank, comm_size, comm->c_name,
                         shm_data->mu_control_block_size));
    OPAL_OUTPUT_VERBOSE((SHARM_LOG_DEBUG, mca_coll_sharm_stream,
                         "coll:sharm: comm (%d/%d/%s): "
                         "mu_pids_block_size is %ld",
                         comm_rank, comm_size, comm->c_name,
                         shm_data->mu_pids_block_size));

    shm_data->mu_seg_size
        = shm_data->mu_barrier_block_size   /* Barrier */
          + shm_data->mu_queue_block_size   /* Queue system  */
          + shm_data->mu_control_block_size /* Queue control system */
          + shm_data->mu_pids_block_size    /* PIDs for CMA algorithms */
#if SHARM_CHECK_XPMEM_SUPPORT
          + shm_data->mu_xpmem_segid_block_size
#endif
        ;

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_INFO, mca_coll_sharm_stream,
                         "coll:sharm: comm (%d/%d/%s): "
                         "attaching to %" PRIsize_t " byte mmap: %s",
                         comm_rank, comm_size, comm->c_name,
                         shm_data->mu_seg_size, fullpath));

    if (0 == comm_rank) {
        /*
         * FIXME: Extra ps for avoid segfault when we write into
         * first/last(?) page when seg_size has lowest value.
         */

        shm_data->segmeta = mca_common_sm_module_create_and_attach(
            shm_data->mu_seg_size + ps, fullpath,
            sizeof(mca_common_sm_seg_header_t), ps);

        if (NULL == shm_data->segmeta) {
            opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                                "coll:sharm: comm (%d/%d/%s): "
                                "mca_common_sm_module_create_and_attach failed",
                                comm_rank, comm_size, comm->c_name);
            free(fullpath);
            return OMPI_ERR_OUT_OF_RESOURCE;
        }

        memset(shm_data->segmeta->module_data_addr, 0, shm_data->mu_seg_size);

        for (int i = 1; i < comm_size; i++) {
            MCA_PML_CALL(send(&shm_data->segmeta->shmem_ds,
                              sizeof(shm_data->segmeta->shmem_ds), MPI_BYTE, i,
                              MCA_COLL_BASE_TAG_BCAST,
                              MCA_PML_BASE_SEND_STANDARD, comm));
        }
    } else {
        opal_shmem_ds_t ds;
        MCA_PML_CALL(recv(&ds, sizeof(ds), MPI_BYTE, 0, MCA_COLL_BASE_TAG_BCAST,
                          comm, MPI_STATUS_IGNORE));
        shm_data->segmeta = mca_common_sm_module_attach(
            &ds, sizeof(mca_common_sm_seg_header_t), ps);
    }

    free(fullpath);

    return OMPI_SUCCESS;
}
