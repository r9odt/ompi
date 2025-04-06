/** @file */

#include "coll_sharm.h"

/*
 * External variables.
 */

extern int mca_coll_sharm_stream;
extern int mca_coll_sharm_stream;

/**
 * @brief check is communicator processes share a same compute node.
 * @param[in] comm mpi communicator.
 * @return SHARM_TRUE if all processes share a same compute node.
 */
int sharm_is_single_node_mode(ompi_communicator_t *comm)
{
    return ompi_group_have_remote_peers(comm->c_local_group) ? SHARM_FALSE
                                                             : SHARM_TRUE;
}

/**
 * @brief check get node leaders.
 * @param[in] comm mpi communicator.
 * @return rank of node leader.
 */
int sharm_process_topology(mca_coll_sharm_module_t *module)
{
    int comm_rank, ret, local_rank;
    int values[2] = {0, 0};
    ompi_communicator_t *comm = module->comm;

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_CALL, mca_coll_sharm_stream,
                         "coll:sharm:sharm_process_topology: (%d/%d/%s) call",
                         ompi_comm_rank(comm), ompi_comm_size(comm),
                         comm->c_name));
    OPAL_OUTPUT_VERBOSE((SHARM_LOG_INFO, mca_coll_sharm_stream,
                         "coll:sharm:sharm_process_topology: (%d/%d/%s) "
                         "my_node_rank: %d my_numa_rank: %d",
                         ompi_comm_rank(comm), ompi_comm_size(comm),
                         comm->c_name, opal_process_info.my_node_rank,
                         opal_process_info.my_numa_rank));
    if (NULL != opal_hwloc_topology) {
        int ncores = hwloc_get_nbobjs_by_type(opal_hwloc_topology,
                                              HWLOC_OBJ_CORE);
        int nnuma = hwloc_get_nbobjs_by_type(opal_hwloc_topology,
                                             HWLOC_OBJ_NUMANODE);
        int npackages = hwloc_get_nbobjs_by_type(opal_hwloc_topology,
                                                 HWLOC_OBJ_PACKAGE);
        int nmachines = hwloc_get_nbobjs_by_type(opal_hwloc_topology,
                                                 HWLOC_OBJ_MACHINE);

        opal_output_verbose(SHARM_LOG_INFO, mca_coll_sharm_stream,
                            "coll:sharm:sharm_process_topology: (%d/%d/%s) "
                            "ncores %d, nnuma %d, "
                            "npack %d, nmachines %d",
                            ompi_comm_rank(comm), ompi_comm_size(comm),
                            comm->c_name, ncores, nnuma, npackages, nmachines);
    }
    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_CALL, mca_coll_sharm_stream,
                         "coll:sharm:opal_hwloc_topology: (%d/%d/%s) checked",
                         ompi_comm_rank(comm), ompi_comm_size(comm),
                         comm->c_name));

    ret = ompi_comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, NULL,
                               &module->shared_comm);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != ret)) {
        opal_output_verbose(
            SHARM_LOG_ERROR, mca_coll_sharm_stream,
            "coll:sharm:opal_hwloc_topology: (%d/%d/%s) failed to create a "
            "shared memory communicator. error code %d",
            ompi_comm_rank(comm), ompi_comm_size(comm), comm->c_name, ret);
        return ret;
    }
    local_rank = ompi_comm_rank(module->shared_comm);
    comm_rank = ompi_comm_rank(module->comm);
    ret = ompi_comm_split(module->comm, (0 == local_rank) ? 0 : MPI_UNDEFINED,
                          comm_rank, &module->node_leaders, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != ret)) {
        opal_output_verbose(
            SHARM_LOG_ERROR, mca_coll_sharm_stream,
            "coll:sharm:sharm_process_topology: (%d/%d/%s) failed to create a "
            "local leaders communicator. error code %d",
            ompi_comm_rank(comm), ompi_comm_size(comm), comm->c_name, ret);
        return ret;
    }
    if (0 == local_rank) {
        values[0] = ompi_comm_size(module->node_leaders);
        values[1] = ompi_comm_rank(module->node_leaders);
        opal_output_verbose(SHARM_LOG_INFO, mca_coll_sharm_stream,
                            "coll:sharm:sharm_process_topology: (%d/%d/%s) "
                            "local leaders communicator. rank: %d size: %d",
                            ompi_comm_rank(comm), ompi_comm_size(comm),
                            comm->c_name, values[1], values[0]);
    }
    if (ompi_comm_size(module->shared_comm) > 1) {
        ret = module->shared_comm->c_coll
                  ->coll_bcast(values, 2, MPI_INT, 0, module->shared_comm,
                               module->shared_comm->c_coll->coll_bcast_module);
    }
    if (OMPI_SUCCESS != ret) {
        opal_output_verbose(SHARM_LOG_ERROR, mca_coll_sharm_stream,
                            "coll:sharm:sharm_process_topology: (%d/%d/%s) "
                            "failed to broadcast local data. error code %d",
                            ompi_comm_rank(comm), ompi_comm_size(comm),
                            comm->c_name, ret);
        return ret;
    }

    module->node_count = values[0];
    module->node_id = values[1];
    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_CALL, mca_coll_sharm_stream,
                         "coll:sharm:sharm_process_topology: (%d/%d/%s) check "
                         "done. Node id %d/%d",
                         ompi_comm_rank(comm), ompi_comm_size(comm),
                         comm->c_name, module->node_id, module->node_count));
    return OMPI_SUCCESS;
}

/**
 * @brief calculate memory pages required for storing count elems of size bytes.
 * @param[in] count count of elements.
 * @param[in] size size of one element.
 * @return number of memory pages required
 * for storing count elems of size bytes.
 */
int sharm_get_npages(size_t count, size_t size)
{
    size_t nbytes = count * size;
    int ps = opal_getpagesize();
    return (nbytes + ps - 1) / ps; /* ceil(nbytes/ps) */
}

/*
 * sharm_is_page_aligned: Returns 1 if addr is aligned to the page boundary
 */
/**
 * @brief check addr is aligned to the page boundary.
 * @param[in] addr address to check.
 * @return 1 if addr is aligned to the page boundary.
 */
int sharm_is_page_aligned(void *addr)
{
    size_t ps = opal_getpagesize();
    return ((ptrdiff_t) (addr) & (ps - 1)) == 0;
}

#if SHARM_CHECK_NUMA_SUPPORT
int sharm_move_page_to_numa_node(void *addr)
{
    return 0;
    /* Addr must be page aligned */
    // void *addr_aligned[1] = {addr};
    // int status[1] = {-1};
    // if (!sharm_is_page_aligned(addr_aligned[0]))
    //     return -1;
    // move_pages(0, 1, addr_aligned, NULL, status, 0);
    // return status[0];
}

/*
 * sharm_check_pages_mem_affinity
 */
void sharm_check_pages_mem_affinity(int rank, void *addr, int size,
                                    char *blockname)
{
    return;
    // int c, s;
    // s = getcpu(&c, NULL, NULL);
    // int cpu = (s == -1) ? s : c;
    // if (cpu < 0) {
    //     opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
    //                         "coll:sharm: error of determinate cpu");
    //     return;
    // }

    // int cpu = sched_getcpu();
    // int proc_numa_node = numa_node_of_cpu(cpu);
    // if (proc_numa_node < 0) {
    //     return;
    // }
    // int ps = opal_getpagesize();
    // int npages = sharm_get_npages(1, size);
    // for (int i = 0; i < npages; i++) {
    //     int page_numa_node = sharm_move_page_to_numa_node((char *) addr
    //                                                       + i * ps);
    //     if (page_numa_node != proc_numa_node) {
    //         OPAL_OUTPUT_VERBOSE(
    //             (SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
    //              "coll:sharm: memory affinity error: rank %d on numa node %d:
    //              "
    //              "%s page %d on node %d",
    //              rank, proc_numa_node, blockname, i, page_numa_node));
    //     }
    // }
}
#else
int sharm_move_page_to_numa_node(void *addr)
{
    return 0;
}

void sharm_check_pages_mem_affinity(int rank, void *addr, int size,
                                    char *blockname)
{
}
#endif

#if SHARM_CHECK_CMA_SUPPORT
/* sharm_cma_readv: Copy size bytes from remote_base to local_base */
int sharm_cma_readv(pid_t remote_pid, void *local_base, void *remote_base,
                    size_t size)
{
    struct iovec src_iov = {.iov_base = remote_base, .iov_len = size};
    struct iovec dst_iov = {.iov_base = local_base, .iov_len = size};

    do {
        ssize_t nbytes = process_vm_readv(remote_pid, &dst_iov, 1, &src_iov, 1,
                                          0);
        if (nbytes < 0) {
            return 0;
        }
        src_iov.iov_base = (void *) ((char *) src_iov.iov_base + nbytes);
        src_iov.iov_len -= nbytes;
        dst_iov.iov_base = (void *) ((char *) dst_iov.iov_base + nbytes);
        dst_iov.iov_len -= nbytes;
    } while (0 < src_iov.iov_len);

    return size;
}
/* sharm_cma_writev: Copy size bytes from local_base to remote_base*/
int sharm_cma_writev(pid_t remote_pid, void *local_base, void *remote_base,
                     size_t size)
{
    struct iovec src_iov = {.iov_base = local_base, .iov_len = size};
    struct iovec dst_iov = {.iov_base = remote_base, .iov_len = size};

    do {
        ssize_t nbytes = process_vm_writev(remote_pid, &src_iov, 1, &dst_iov, 1,
                                           0);
        if (nbytes < 0) {
            return 0;
        }
        src_iov.iov_base = (void *) ((char *) src_iov.iov_base + nbytes);
        src_iov.iov_len -= nbytes;
        dst_iov.iov_base = (void *) ((char *) dst_iov.iov_base + nbytes);
        dst_iov.iov_len -= nbytes;
    } while (0 < src_iov.iov_len);

    return size;
}
#else
int sharm_cma_readv(pid_t remote_pid, void *local_base, void *remote_base,
                    size_t size)
{
    return -1;
}
int sharm_cma_writev(pid_t remote_pid, void *local_base, void *remote_base,
                     size_t size)
{
    return -1;
}
#endif