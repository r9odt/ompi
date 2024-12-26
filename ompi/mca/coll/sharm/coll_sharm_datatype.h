/** @file */

#ifndef MCA_COLL_SHARM_DATATYPE_H
#define MCA_COLL_SHARM_DATATYPE_H

#include "ompi_config.h"
#include "mpi.h"

#include "coll_sharm_util.h"

#include "ompi/communicator/communicator.h"
#include "ompi/constants.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/mca/coll/base/base.h"
#include "opal/mca/common/sm/common_sm.h"

/**
 * @brief Sharm component structure.
 */
typedef struct mca_coll_sharm_component_t {
    /*
     * Base coll component.
     */
    mca_coll_base_component_t super;

    /*
     * MCA parameter: Priority of this component.
     */
    int priority;
} mca_coll_sharm_component_t;

/**
 * @brief Sharm shared memory component control structure.
 */
typedef struct sharm_coll_data_t {
    /* Shared-memory segment descriptor. */
    mca_common_sm_module_t *segmeta;

    uint32_t nproc; // Count of processes for this segment

    /* Barrier vars. */
    uint32_t barrier_sr_pvt_sense;

    uint32_t *current_queue_slots; // Counters for current slot in queue

    /* Memory parameters */

    size_t mu_page_size;      // Size of one cache page.
    size_t mu_cacheline_size; // Size of one cache line.

    size_t mu_seg_size; // All segment size.

    size_t mu_barrier_block_size; // Barrier block size.

    size_t mu_queue_nfrags; // Count of fragments.

    size_t mu_queue_block_size;            // All queue size.
    size_t mu_queue_one_queue_size;        // One queue size.
    size_t mu_queue_one_queue_system_size; // One queue system size.
    size_t mu_queue_fragment_size;         // Size of one fragment.

    size_t mu_control_block_size;            // All queue control size.
    size_t mu_control_one_queue_size;        // One queue control size.
    size_t mu_control_one_queue_system_size; // One queue system control size.
    size_t mu_control_fragment_size;         // One control fragment size.

    size_t mu_pids_block_size; // Process ID storage size.

#if SHARM_CHECK_XPMEM_SUPPORT
    size_t mu_xpmem_segid_block_size; // XPMEM Seg ID storage size
#endif

    /* Pointers to shared memory blocks. */

    /* Barrier shared fields */
    uint32_t *shm_barrier_sr_sense;
    /* Sense-reversing barrier */
    opal_atomic_uint32_t *shm_barrier_sr_counter;

    /* XPMEM shared Fields */

#if SHARM_CHECK_XPMEM_SUPPORT
    /* XPMEM segment IDs for each rank */
    xpmem_segid_t **xpmem_segid;
#endif

    /* Queue system */
    unsigned char **shm_process_ids;
    unsigned char **shm_queue;
    unsigned char **shm_control;
} sharm_coll_data_t;

#define GLOBAL_PROFILING_COUNTERS(operation)               \
    double push_global_time_##operation;                   \
    double pop_global_time_##operation;                    \
    double copy_global_time_##operation;                   \
    double reduce_operation_global_time_##operation;       \
    double total_global_time_##operation;                  \
    double xpmem_attach_global_time_##operation;           \
    double collective_exchange_global_time_##operation;    \
    double zcopy_barrier_global_time_##operation;          \
    uint64_t push_global_count_##operation;                \
    uint64_t pop_global_count_##operation;                 \
    uint64_t copy_global_count_##operation;                \
    uint64_t reduce_operation_global_count_##operation;    \
    uint64_t total_global_count_##operation;               \
    uint64_t xpmem_attach_global_count_##operation;        \
    uint64_t collective_exchange_global_count_##operation; \
    uint64_t zcopy_barrier_global_count_##operation;

/**
 * @brief Sharm profiling flags and counters.
 */
typedef struct mca_coll_sharm_profiling_data_t {
    GLOBAL_PROFILING_COUNTERS(barrier)
    GLOBAL_PROFILING_COUNTERS(bcast)
    GLOBAL_PROFILING_COUNTERS(gather)
    GLOBAL_PROFILING_COUNTERS(gatherv)
    GLOBAL_PROFILING_COUNTERS(scatter)
    GLOBAL_PROFILING_COUNTERS(scatterv)
    GLOBAL_PROFILING_COUNTERS(alltoall)
    GLOBAL_PROFILING_COUNTERS(alltoallv)
    GLOBAL_PROFILING_COUNTERS(alltoallw)
    GLOBAL_PROFILING_COUNTERS(allgather)
    GLOBAL_PROFILING_COUNTERS(allgatherv)
    GLOBAL_PROFILING_COUNTERS(scan)
    GLOBAL_PROFILING_COUNTERS(exscan)
    GLOBAL_PROFILING_COUNTERS(reduce)
    GLOBAL_PROFILING_COUNTERS(reduce_scatter)
    GLOBAL_PROFILING_COUNTERS(reduce_scatter_block)
    GLOBAL_PROFILING_COUNTERS(allreduce)
} mca_coll_sharm_profiling_data_t;

#define RESOLVE_COLLECTIVIES_DATA(module, rank)     \
    (char *) module->local_collectivies_info.memory \
        + rank * module->local_collectivies_info.one_rank_block_size

/**
 * @brief Sharm collectivies exchange data structure.
 */
typedef struct sharm_local_collectivies_data_t {
    size_t one_rank_block_size;

    // collectivies data
    void *memory;

    // memory for store pointers
    void *pointers_memory;

    /*
     * Base pointers
     */

    ptrdiff_t **sbuf;
    ptrdiff_t **rbuf;
    ptrdiff_t **sdtypes_ext;
    ptrdiff_t **rdtypes_ext;
    size_t **scounts;
    size_t **rcounts;
    size_t **sdtypes_size;
    size_t **rdtypes_size;
    char **sdtypes_contiguous;
    char **rdtypes_contiguous;
} sharm_local_collectivies_data_t;

typedef struct mca_coll_sharm_fallbacks_t {
    mca_coll_base_module_barrier_fn_t fallback_barrier;
    mca_coll_base_module_t *fallback_barrier_module;
    mca_coll_base_module_bcast_fn_t fallback_bcast;
    mca_coll_base_module_t *fallback_bcast_module;
    mca_coll_base_module_scatter_fn_t fallback_scatter;
    mca_coll_base_module_t *fallback_scatter_module;
    mca_coll_base_module_scatterv_fn_t fallback_scatterv;
    mca_coll_base_module_t *fallback_scatterv_module;
    mca_coll_base_module_gather_fn_t fallback_gather;
    mca_coll_base_module_t *fallback_gather_module;
    mca_coll_base_module_gatherv_fn_t fallback_gatherv;
    mca_coll_base_module_t *fallback_gatherv_module;
    mca_coll_base_module_allgather_fn_t fallback_allgather;
    mca_coll_base_module_t *fallback_allgather_module;
    mca_coll_base_module_allgatherv_fn_t fallback_allgatherv;
    mca_coll_base_module_t *fallback_allgatherv_module;
    mca_coll_base_module_alltoall_fn_t fallback_alltoall;
    mca_coll_base_module_t *fallback_alltoall_module;
    mca_coll_base_module_alltoallv_fn_t fallback_alltoallv;
    mca_coll_base_module_t *fallback_alltoallv_module;
    mca_coll_base_module_alltoallw_fn_t fallback_alltoallw;
    mca_coll_base_module_t *fallback_alltoallw_module;
    mca_coll_base_module_reduce_scatter_block_fn_t
        fallback_reduce_scatter_block;
    mca_coll_base_module_t *fallback_reduce_scatter_block_module;
    mca_coll_base_module_reduce_scatter_fn_t fallback_reduce_scatter;
    mca_coll_base_module_t *fallback_reduce_scatter_module;
    mca_coll_base_module_reduce_fn_t fallback_reduce;
    mca_coll_base_module_t *fallback_reduce_module;
    mca_coll_base_module_allreduce_fn_t fallback_allreduce;
    mca_coll_base_module_t *fallback_allreduce_module;
    mca_coll_base_module_scan_fn_t fallback_scan;
    mca_coll_base_module_t *fallback_scan_module;
    mca_coll_base_module_exscan_fn_t fallback_exscan;
    mca_coll_base_module_t *fallback_exscan_module;
} mca_coll_sharm_fallbacks_t;

/**
 * @brief Sharm module definition.
 */
typedef struct mca_coll_sharm_module_t {
    mca_coll_base_module_t super;

    /* Local memory map to collective info excange. */
    sharm_local_collectivies_data_t local_collectivies_info;

    /*
     * Profiling data.
     */
    mca_coll_sharm_profiling_data_t profiling_data;

    /* Operation number. */
    uint32_t local_op;
    /* Barrier number. */
    uint32_t local_barrier;
    /* Bcast number. */
    uint32_t local_bcast;
    /* Gather number. */
    uint32_t local_gather;
    /* Gatherv number. */
    uint32_t local_gatherv;
    /* Scatter number. */
    uint32_t local_scatter;
    /* Scatterv number. */
    uint32_t local_scatterv;
    /* Scan number. */
    uint32_t local_scan;
    /* Exscan number. */
    uint32_t local_exscan;
    /* Allreduce number. */
    uint32_t local_allreduce;
    /* Reduce number. */
    uint32_t local_reduce;
    /* Reduce scatter number. */
    uint32_t local_reduce_scatter;
    /* Reduce scatter block number. */
    uint32_t local_reduce_scatter_block;
    /* Allgather number. */
    uint32_t local_allgather;
    /* Allgatherv number. */
    uint32_t local_allgatherv;
    /* Alltoall number. */
    uint32_t local_alltoall;
    /* Alltoallv number. */
    uint32_t local_alltoallv;
    /* Alltoallw number. */
    uint32_t local_alltoallw;

#if SHARM_CHECK_XPMEM_SUPPORT
    /*
     * XPMEM APIDs for each rank
     */
    xpmem_apid_t *apids;
    char xpmem_runtime_check_support;
#endif

    /*
     * Module communicator.
     */
    ompi_communicator_t *comm;

    /*
     * Shared-memory data.
     */
    sharm_coll_data_t *shared_memory_data;

    /* Local memory map to support operations. */
    void *local_op_memory_map;

    /*
     * Fallback links.
     */
    mca_coll_sharm_fallbacks_t fallbacks;
} mca_coll_sharm_module_t;

/**
 * @brief Declare mca_coll_sharm_module_t_class.
 */
OBJ_CLASS_DECLARATION(mca_coll_sharm_module_t);

#endif /* MCA_COLL_SHARM_DATATYPE_H */
