/** @file */

#ifndef MCA_COLL_SHARM_FUNC_H
#define MCA_COLL_SHARM_FUNC_H

#include "coll_sharm_datatype.h"
#include "coll_sharm_profiling.h"

#include "ompi_config.h"
#include "ompi/communicator/communicator.h"
#include "ompi/datatype/ompi_datatype.h"

/*
 * Macros.
 */

#define SHARM_INIT()                                                          \
    if (OPAL_UNLIKELY(                                                        \
            (NULL                                                             \
             == ((mca_coll_sharm_module_t *) module)->shared_memory_data))) { \
        int err = OMPI_SUCCESS;                                               \
        if ((err = mca_coll_sharm_init_segment(module)) != OMPI_SUCCESS) {    \
            return err;                                                       \
        };                                                                    \
    }

#define SHARM_NEW_OP(module) module->local_op++

#define SHARM_OP(module) module->local_op

#define SHARM_NEW_COLL(coll, module) module->local_##coll++

#define SHARM_COLL(coll, module) module->local_##coll

/**
 * @brief Resolve PID for rank process.
 * @param[in] shm_data Pointer to shared memory structure.
 * @param[in] rank process ramk.
 * @return Process ID.
 */
#define SHARM_GET_RANK_PID(shm_data, rank) \
    *((pid_t *) (shm_data->shm_process_ids[rank]))

#define SHARM_SPIN_CONDITION_MAX 1000000
#define SHARM_SPIN_CONDITION(cond)                                     \
    {                                                                  \
        __label__ exit_label;                                          \
        do {                                                           \
            if (cond)                                                  \
                goto exit_label;                                       \
            for (int __i = 0; __i < SHARM_SPIN_CONDITION_MAX; __i++) { \
                if (cond) {                                            \
                    goto exit_label;                                   \
                }                                                      \
            }                                                          \
            opal_progress();                                           \
        } while (1);                                                   \
    exit_label:;                                                       \
    }

#define SHRAM_DUMP_CURRENT_SLOTS(shm_data, comm)                             \
    for (int __i = 0; __i < ompi_comm_size(comm); ++__i) {                   \
        for (int __j = 0; __j < ompi_comm_size(comm); ++__j) {               \
            OPAL_OUTPUT_VERBOSE(                                             \
                (SHARM_LOG_TRACE, mca_coll_sharm_stream,                     \
                 "coll:sharm:%d:dump: (%d/%d/%s) current_slot[%d][%d] = %d", \
                 SHARM_OP(sharm_module), ompi_comm_rank(comm),               \
                 ompi_comm_size(comm), comm->c_name, __i, __j,               \
                 SHARM_CURRENT_SLOT_RESOLVE(shm_data, __i, __j)));           \
        }                                                                    \
    }

#if !defined(min)
#    define min(a, b) (a < b) ? a : b
#endif

/*
 * Functions.
 */

int mca_coll_sharm_init_segment(mca_coll_base_module_t *module);
int sharm_is_single_node_mode(ompi_communicator_t *comm);
int sharm_get_npages(size_t count, size_t size);
int sharm_allocate_segment(mca_coll_base_module_t *module);
int sharm_is_page_aligned(void *addr);
int sharm_cma_readv(pid_t remote_pid, void *local_base, void *remote_base,
                    size_t size);
int sharm_cma_writev(pid_t remote_pid, void *local_base, void *remote_base,
                     size_t size);

int sharm_barrier_gather_cico(int root, ompi_communicator_t *comm,
                              mca_coll_base_module_t *module);
int sharm_barrier_bcast_cico(int root, ompi_communicator_t *comm,
                             mca_coll_base_module_t *module);

// TODO:

int sharm_process_topology(ompi_communicator_t *comm);
int sharm_move_page_to_numa_node(void *addr);
void sharm_check_pages_mem_affinity(int rank, void *addr, int npages,
                                    char *blockname);

/*
 * Collectives.
 */

int sharm_barrier_intra(ompi_communicator_t *comm,
                        mca_coll_base_module_t *module);
int sharm_barrier_sense_reversing(ompi_communicator_t *comm,
                                  mca_coll_base_module_t *module);
int sharm_barrier_cico(ompi_communicator_t *comm,
                       mca_coll_base_module_t *module);
int sharm_barrier_bcast_cico(int root, ompi_communicator_t *comm,
                             mca_coll_base_module_t *module);

int sharm_bcast_intra(void *buff, int count, ompi_datatype_t *datatype,
                      int root, ompi_communicator_t *comm,
                      mca_coll_base_module_t *module);
int sharm_bcast_cico(void *buff, int count, ompi_datatype_t *datatype, int root,
                     ompi_communicator_t *comm, mca_coll_base_module_t *module);
int sharm_bcast_cma(void *buff, int count, ompi_datatype_t *datatype, int root,
                    ompi_communicator_t *comm, mca_coll_base_module_t *module);
int sharm_bcast_xpmem(void *buff, int count, ompi_datatype_t *datatype,
                      int root, ompi_communicator_t *comm,
                      mca_coll_base_module_t *module);

int sharm_scatter_intra(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                        void *rbuf, int rcount, ompi_datatype_t *rdtype,
                        int root, ompi_communicator_t *comm,
                        mca_coll_base_module_t *module);
int sharm_scatter_cico(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                       void *rbuf, int rcount, ompi_datatype_t *rdtype,
                       int root, ompi_communicator_t *comm,
                       mca_coll_base_module_t *module);
int sharm_scatter_cma(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                      void *rbuf, int rcount, ompi_datatype_t *rdtype, int root,
                      ompi_communicator_t *comm,
                      mca_coll_base_module_t *module);
int sharm_scatter_xpmem(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                        void *rbuf, int rcount, ompi_datatype_t *rdtype,
                        int root, ompi_communicator_t *comm,
                        mca_coll_base_module_t *module);

int sharm_scatterv_intra(const void *sbuf, const int *scounts,
                         const int *displs, ompi_datatype_t *sdtype, void *rbuf,
                         int rcount, ompi_datatype_t *rdtype, int root,
                         ompi_communicator_t *comm,
                         mca_coll_base_module_t *module);
int sharm_scatterv_cico(const void *sbuf, const int *scounts, const int *displs,
                        ompi_datatype_t *sdtype, void *rbuf, int rcount,
                        ompi_datatype_t *rdtype, int root,
                        ompi_communicator_t *comm,
                        mca_coll_base_module_t *module);
int sharm_scatterv_cma(const void *sbuf, const int *scounts, const int *displs,
                       ompi_datatype_t *sdtype, void *rbuf, int rcount,
                       ompi_datatype_t *rdtype, int root,
                       ompi_communicator_t *comm,
                       mca_coll_base_module_t *module);
int sharm_scatterv_xpmem(const void *sbuf, const int *scounts,
                         const int *displs, ompi_datatype_t *sdtype, void *rbuf,
                         int rcount, ompi_datatype_t *rdtype, int root,
                         ompi_communicator_t *comm,
                         mca_coll_base_module_t *module);

int sharm_gather_intra(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                       void *rbuf, int rcount, ompi_datatype_t *rdtype,
                       int root, ompi_communicator_t *comm,
                       mca_coll_base_module_t *module);
int sharm_gather_cico(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                      void *rbuf, int rcount, ompi_datatype_t *rdtype, int root,
                      ompi_communicator_t *comm,
                      mca_coll_base_module_t *module);
int sharm_gather_cma(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                     void *rbuf, int rcount, ompi_datatype_t *rdtype, int root,
                     ompi_communicator_t *comm, mca_coll_base_module_t *module);
int sharm_gather_xpmem(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                       void *rbuf, int rcount, ompi_datatype_t *rdtype,
                       int root, ompi_communicator_t *comm,
                       mca_coll_base_module_t *module);

int sharm_gatherv_intra(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                        void *rbuf, const int *rcounts, const int *displs,
                        ompi_datatype_t *rdtype, int root,
                        ompi_communicator_t *comm,
                        mca_coll_base_module_t *module);
int sharm_gatherv_cico(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                       void *rbuf, const int *rcounts, const int *displs,
                       ompi_datatype_t *rdtype, int root,
                       ompi_communicator_t *comm,
                       mca_coll_base_module_t *module);
int sharm_gatherv_cma(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                      void *rbuf, const int *rcounts, const int *displs,
                      ompi_datatype_t *rdtype, int root,
                      ompi_communicator_t *comm,
                      mca_coll_base_module_t *module);
int sharm_gatherv_xpmem(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                        void *rbuf, const int *rcounts, const int *displs,
                        ompi_datatype_t *rdtype, int root,
                        ompi_communicator_t *comm,
                        mca_coll_base_module_t *module);

int sharm_allgatherv_intra(const void *sbuf, int scount,
                           ompi_datatype_t *sdtype, void *rbuf,
                           const int *rcounts, const int *displs,
                           ompi_datatype_t *rdtype, ompi_communicator_t *comm,
                           mca_coll_base_module_t *module);
int sharm_allgatherv_cico(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                          void *rbuf, const int *rcounts, const int *displs,
                          ompi_datatype_t *rdtype, ompi_communicator_t *comm,
                          mca_coll_base_module_t *module);
int sharm_allgatherv_cma(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                         void *rbuf, const int *rcounts, const int *displs,
                         ompi_datatype_t *rdtype, ompi_communicator_t *comm,
                         mca_coll_base_module_t *module);
int sharm_allgatherv_xpmem(const void *sbuf, int scount,
                           ompi_datatype_t *sdtype, void *rbuf,
                           const int *rcounts, const int *displs,
                           ompi_datatype_t *rdtype, ompi_communicator_t *comm,
                           mca_coll_base_module_t *module);

int sharm_allgather_intra(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                          void *rbuf, int rcount, ompi_datatype_t *rdtype,
                          ompi_communicator_t *comm,
                          mca_coll_base_module_t *module);
int sharm_allgather_cico(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                         void *rbuf, int rcount, ompi_datatype_t *rdtype,
                         ompi_communicator_t *comm,
                         mca_coll_base_module_t *module);
int sharm_allgather_cma(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                        void *rbuf, int rcount, ompi_datatype_t *rdtype,
                        ompi_communicator_t *comm,
                        mca_coll_base_module_t *module);
int sharm_allgather_xpmem(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                          void *rbuf, int rcount, ompi_datatype_t *rdtype,
                          ompi_communicator_t *comm,
                          mca_coll_base_module_t *module);

int sharm_reduce_intra(const void *sbuf, void *rbuf, int count,
                       ompi_datatype_t *dtype, struct ompi_op_t *op, int root,
                       struct ompi_communicator_t *comm,
                       mca_coll_base_module_t *module);
int sharm_reduce_cico_knomial(const void *sbuf, void *rbuf, int count,
                              ompi_datatype_t *dtype, struct ompi_op_t *op,
                              int root, struct ompi_communicator_t *comm,
                              mca_coll_base_module_t *module);
int sharm_reduce_cico_non_commutative(const void *sbuf, void *rbuf, int count,
                                      ompi_datatype_t *dtype,
                                      struct ompi_op_t *op, int root,
                                      struct ompi_communicator_t *comm,
                                      mca_coll_base_module_t *module);
int sharm_reduce_cma(const void *sbuf, void *rbuf, int count,
                     ompi_datatype_t *dtype, struct ompi_op_t *op, int root,
                     struct ompi_communicator_t *comm,
                     mca_coll_base_module_t *module);
int sharm_reduce_xpmem(const void *sbuf, void *rbuf, int count,
                       ompi_datatype_t *dtype, struct ompi_op_t *op, int root,
                       struct ompi_communicator_t *comm,
                       mca_coll_base_module_t *module);

int sharm_allreduce_intra(const void *sbuf, void *rbuf, int count,
                          ompi_datatype_t *dtype, struct ompi_op_t *op,
                          struct ompi_communicator_t *comm,
                          mca_coll_base_module_t *module);

int sharm_allreduce_cico_non_commutative(const void *sbuf, void *rbuf,
                                         int count, ompi_datatype_t *dtype,
                                         struct ompi_op_t *op,
                                         struct ompi_communicator_t *comm,
                                         mca_coll_base_module_t *module);
int sharm_allreduce_cico_reduce_broadcast(const void *sbuf, void *rbuf,
                                          int count, ompi_datatype_t *dtype,
                                          struct ompi_op_t *op,
                                          struct ompi_communicator_t *comm,
                                          mca_coll_base_module_t *module);

int sharm_alltoall_intra(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                         void *rbuf, int rcount, ompi_datatype_t *rdtype,
                         struct ompi_communicator_t *comm,
                         mca_coll_base_module_t *module);
int sharm_alltoall_cico(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                        void *rbuf, int rcount, ompi_datatype_t *rdtype,
                        struct ompi_communicator_t *comm,
                        mca_coll_base_module_t *module);
int sharm_alltoall_cma(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                       void *rbuf, int rcount, ompi_datatype_t *rdtype,
                       struct ompi_communicator_t *comm,
                       mca_coll_base_module_t *module);
int sharm_alltoall_xpmem(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                         void *rbuf, int rcount, ompi_datatype_t *rdtype,
                         struct ompi_communicator_t *comm,
                         mca_coll_base_module_t *module);

int sharm_alltoallv_intra(const void *sbuf, const int *scounts,
                          const int *sdispls, ompi_datatype_t *sdtype,
                          void *rbuf, const int *rcounts, const int *rdispls,
                          ompi_datatype_t *rdtype,
                          struct ompi_communicator_t *comm,
                          mca_coll_base_module_t *module);
int sharm_alltoallv_cico(const void *sbuf, const int *scounts,
                         const int *sdispls, ompi_datatype_t *sdtype,
                         void *rbuf, const int *rcounts, const int *rdispls,
                         ompi_datatype_t *rdtype,
                         struct ompi_communicator_t *comm,
                         mca_coll_base_module_t *module);
int sharm_alltoallv_cma(const void *sbuf, const int *scounts,
                        const int *sdispls, ompi_datatype_t *sdtype, void *rbuf,
                        const int *rcounts, const int *rdispls,
                        ompi_datatype_t *rdtype,
                        struct ompi_communicator_t *comm,
                        mca_coll_base_module_t *module);
int sharm_alltoallv_xpmem(const void *sbuf, const int *scounts,
                          const int *sdispls, ompi_datatype_t *sdtype,
                          void *rbuf, const int *rcounts, const int *rdispls,
                          ompi_datatype_t *rdtype,
                          struct ompi_communicator_t *comm,
                          mca_coll_base_module_t *module);

int sharm_alltoallw_intra(const void *sbuf, const int *scounts,
                          const int *sdispls, ompi_datatype_t *const *sdtypes,
                          void *rbuf, const int *rcounts, const int *rdispls,
                          ompi_datatype_t *const *rdtypes,
                          struct ompi_communicator_t *comm,
                          mca_coll_base_module_t *module);
int sharm_alltoallw_cico(const void *sbuf, const int *scounts,
                         const int *sdispls, ompi_datatype_t *const *sdtypes,
                         void *rbuf, const int *rcounts, const int *rdispls,
                         ompi_datatype_t *const *rdtypes,
                         struct ompi_communicator_t *comm,
                         mca_coll_base_module_t *module);
int sharm_alltoallw_cma(const void *sbuf, const int *scounts,
                        const int *sdispls, ompi_datatype_t *const *sdtypes,
                        void *rbuf, const int *rcounts, const int *rdispls,
                        ompi_datatype_t *const *rdtypes,
                        struct ompi_communicator_t *comm,
                        mca_coll_base_module_t *module);
int sharm_alltoallw_xpmem(const void *sbuf, const int *scounts,
                          const int *sdispls, ompi_datatype_t *const *sdtypes,
                          void *rbuf, const int *rcounts, const int *rdispls,
                          ompi_datatype_t *const *rdtypes,
                          struct ompi_communicator_t *comm,
                          mca_coll_base_module_t *module);

int sharm_reduce_scatter_intra(const void *sbuf, void *rbuf, const int *rcounts,
                               ompi_datatype_t *dtype, struct ompi_op_t *op,
                               struct ompi_communicator_t *comm,
                               mca_coll_base_module_t *module);
int sharm_reduce_scatter_cico(const void *sbuf, void *rbuf, const int *rcounts,
                              struct ompi_datatype_t *dtype,
                              struct ompi_op_t *op,
                              struct ompi_communicator_t *comm,
                              mca_coll_base_module_t *module);

int sharm_reduce_scatter_block_intra(const void *sbuf, void *rbuf, int rcount,
                                     ompi_datatype_t *dtype,
                                     struct ompi_op_t *op,
                                     struct ompi_communicator_t *comm,
                                     mca_coll_base_module_t *module);
int sharm_reduce_scatter_block_cico(const void *sbuf, void *rbuf, int rcount,
                                    struct ompi_datatype_t *dtype,
                                    struct ompi_op_t *op,
                                    struct ompi_communicator_t *comm,
                                    mca_coll_base_module_t *module);

int sharm_scan_intra(const void *sbuf, void *rbuf, int count,
                     ompi_datatype_t *dtype, struct ompi_op_t *op,
                     struct ompi_communicator_t *comm,
                     mca_coll_base_module_t *module);
int sharm_scan_cico(const void *sbuf, void *rbuf, int count,
                    ompi_datatype_t *dtype, struct ompi_op_t *op,
                    struct ompi_communicator_t *comm,
                    mca_coll_base_module_t *module);

int sharm_exscan_intra(const void *sbuf, void *rbuf, int count,
                       ompi_datatype_t *dtype, struct ompi_op_t *op,
                       struct ompi_communicator_t *comm,
                       mca_coll_base_module_t *module);
int sharm_exscan_cico(const void *sbuf, void *rbuf, int count,
                      ompi_datatype_t *dtype, struct ompi_op_t *op,
                      struct ompi_communicator_t *comm,
                      mca_coll_base_module_t *module);

#endif /* MCA_COLL_SHARM_FUNC_H */
