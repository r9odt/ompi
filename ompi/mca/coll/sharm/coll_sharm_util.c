/** @file */

#include "coll_sharm.h"

/*
 * External variables.
 */

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