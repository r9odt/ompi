/** @file */

#include "coll_sharm.h"

/*
 * External variables.
 */

extern int mca_coll_sharm_stream;
/*
 * Local functions.
 */

/**
 * @brief shared-memory based algorithm for Scatter using CMA ZeroCopy.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_scatter_cma(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                      void *rbuf, int rcount, ompi_datatype_t *rdtype, int root,
                      ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;
    sharm_local_collectivies_data_t *coll_info = &(
        sharm_module->local_collectivies_info);

    int node_comm_rank = ompi_comm_rank(comm);
    int node_comm_size = ompi_comm_size(comm);

    int _rcount = rcount;
    ompi_datatype_t *_rdtype = rdtype;
    char *_rbuf = rbuf;

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
                         "coll:sharm:%d:scatter_cma: (%d/%d/%s) root %d",
                         SHARM_COLL(scatter, sharm_module), node_comm_rank,
                         node_comm_size, comm->c_name, root));

    ptrdiff_t sext;
    ompi_datatype_type_extent(sdtype, &sext);

    /*
     * If rbuf is MPI_IN_PLACE - use data from sbuf.
     */
    if (rbuf == MPI_IN_PLACE) {
        _rcount = scount;
        _rbuf = (char *) sbuf + sext * scount * node_comm_rank;
        _rdtype = sdtype;
    }

    size_t sdtype_size = 0;
    size_t rdtype_size = 0;
    ompi_datatype_type_size(sdtype, &sdtype_size);
    ompi_datatype_type_size(_rdtype, &rdtype_size);

    ptrdiff_t rext;
    ompi_datatype_type_extent(_rdtype, &rext);

    coll_info->sdtypes_contiguous[node_comm_rank][0]
        = ompi_datatype_is_contiguous_memory_layout(_rdtype,
                                                    _rcount * node_comm_size);
    coll_info->sdtypes_ext[node_comm_rank][0] = sext;
    coll_info->rdtypes_ext[node_comm_rank][0] = rext;
    if (root == node_comm_rank) {
        coll_info->sdtypes_contiguous[node_comm_rank][0]
            &= ompi_datatype_is_contiguous_memory_layout(sdtype,
                                                         scount
                                                             * node_comm_size);
        for (int i = 0; i < node_comm_size; ++i) {
            coll_info->sbuf[node_comm_rank][i] = (ptrdiff_t) (((char *) sbuf)
                                                              + sext * scount
                                                                    * i);
        }
        if (rbuf != MPI_IN_PLACE) {
            ompi_datatype_sndrcv((char *) coll_info
                                     ->sbuf[node_comm_rank][node_comm_rank],
                                 scount, sdtype, _rbuf, _rcount, _rdtype);
        }
    }

    void *memory_map = sharm_module->local_op_memory_map;
    size_t *collectivies_info_bytes_received_by_rank = (size_t *) memory_map;

    /*
     * Exchange collectivies info.
     */
    memset(collectivies_info_bytes_received_by_rank, 0,
           node_comm_size * sizeof(size_t));
    size_t collectivies_info_bytes_sended = 0;
    size_t collectivies_info_bytes_received = 0;
    size_t collectivies_info_bytes_total = (node_comm_size - 1)
                                           * coll_info->one_rank_block_size;
    while (collectivies_info_bytes_sended < coll_info->one_rank_block_size
           || collectivies_info_bytes_received
                  < collectivies_info_bytes_total) {
        for (int i = 0; i < node_comm_size; ++i) {
            if (i == node_comm_rank
                || collectivies_info_bytes_received_by_rank[i]
                       >= coll_info->one_rank_block_size) {
                continue;
            }
            int pop = sharm_queue_pop_contiguous(
                RESOLVE_COLLECTIVIES_DATA(sharm_module, i)
                    + collectivies_info_bytes_received_by_rank[i],
                i, comm, sharm_module);
            collectivies_info_bytes_received_by_rank[i] += pop;
            collectivies_info_bytes_received += pop;
        }

        if (collectivies_info_bytes_sended >= coll_info->one_rank_block_size) {
            continue;
        }

        int bytes_to_send = min(coll_info->one_rank_block_size
                                    - collectivies_info_bytes_sended,
                                shm_data->mu_queue_fragment_size);
        SHARM_PROFILING_TIME_START(sharm_module, scatter, push);
        int push = sharm_queue_push_contiguous(
            RESOLVE_COLLECTIVIES_DATA(sharm_module, node_comm_rank)
                + collectivies_info_bytes_sended,
            bytes_to_send, node_comm_rank, -1, comm, sharm_module);
        SHARM_PROFILING_TIME_STOP(sharm_module, scatter, push);
        collectivies_info_bytes_sended += push;
    }

    uint8_t is_contiguous_dtype = 1;
    for (int i = 0; i < node_comm_size; ++i) {
        is_contiguous_dtype &= coll_info->sdtypes_contiguous[i][0];
    }

    if (0 == is_contiguous_dtype) {
        opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                            "coll:sharm:%d:scatter_cma: (%d/%d/%s) "
                            "Unsupported non-contigous datatype",
                            SHARM_COLL(scatter, sharm_module), node_comm_rank,
                            node_comm_size, comm->c_name);
        // TODO: Fallback to cico
        return OMPI_ERR_NOT_SUPPORTED;
    }

    /*
     * Data exchange.
     */

    if (node_comm_rank != root) {
        int64_t total_size = rdtype_size * _rcount;
        int rc = sharm_cma_readv(SHARM_GET_RANK_PID(shm_data, root), _rbuf,
                                 (void *) coll_info->sbuf[root][node_comm_rank],
                                 total_size);
        if (rc != total_size) {
            return OMPI_ERROR;
        }
    }

    opal_atomic_wmb();

    OPAL_OUTPUT_VERBOSE(
        (SHARM_LOG_FUNCTION_INFO, mca_coll_sharm_stream,
         "coll:sharm:%d:scatter_cma: (%d/%d/%s), root %d scatter complete",
         SHARM_COLL(scatter, sharm_module), node_comm_rank, node_comm_size,
         comm->c_name, root));

    return sharm_barrier_gather_cico(root, comm, module);
}
