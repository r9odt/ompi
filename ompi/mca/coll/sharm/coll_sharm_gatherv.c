/** @file */

#include "coll_sharm.h"

/*
 * External variables.
 */

extern int mca_coll_sharm_stream;
extern int mca_coll_sharm_gatherv_algorithm;

/*
 * Local functions.
 */

/**
 * @brief Sharm Gatherv collective operation entrypoint.
 * @param[in] sbuf send buffer.
 * @param[in] scount count of elements to send.
 * @param[in] sdtype datatype to send.
 * @param[in,out] rbuf receive buffer.
 * @param[in] rcounts count of elements for each process in rbuf.
 * @param[in] displs displs for each process in rbuf.
 * Counted in elements of rdtype.
 * @param[in] rdtype datatype to receive.
 * @param[in] root root process for gather operation
 * @param[in] comm communicator for collective operation.
 * @param[in] module sharm module structure.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_gatherv_intra(const void *sbuf, int scount, ompi_datatype_t *sdtype,
                        void *rbuf, const int *rcounts, const int *displs,
                        ompi_datatype_t *rdtype, int root,
                        ompi_communicator_t *comm,
                        mca_coll_base_module_t *module)
{
    SHARM_INIT();
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;

    int ret = OMPI_SUCCESS;

    SHARM_NEW_OP(sharm_module);
    SHARM_NEW_COLL(gatherv, sharm_module);

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_CALL, mca_coll_sharm_stream,
                         "coll:sharm:%d:gatherv: (%d/%d/%s) alg:%d root:%d",
                         SHARM_COLL(gatherv, sharm_module),
                         ompi_comm_rank(comm), ompi_comm_size(comm),
                         comm->c_name, mca_coll_sharm_gatherv_algorithm, root));

    if (!sharm_is_single_node_mode(comm)) {
        opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                            "coll:sharm:%d:gatherv: (%d/%d/%s) "
                            "Operation cannot support multiple nodes, fallback",
                            SHARM_COLL(gatherv, sharm_module),
                            ompi_comm_rank(comm), ompi_comm_size(comm),
                            comm->c_name);
        return sharm_module->fallbacks
            .fallback_gatherv(sbuf, scount, sdtype, rbuf, rcounts, displs,
                              rdtype, root, comm,
                              sharm_module->fallbacks.fallback_gatherv_module);
    }

    switch (mca_coll_sharm_gatherv_algorithm) {
    case COLL_SHARM_GATHERV_ALG_CMA:
#if SHARM_CHECK_CMA_SUPPORT
        SHARM_PROFILING_TOTAL_TIME_START(sharm_module, gatherv);
        ret = sharm_gatherv_cma(sbuf, scount, sdtype, rbuf, rcounts, displs,
                                rdtype, root, comm, module);
        SHARM_PROFILING_TOTAL_TIME_STOP(sharm_module, gatherv);
        return ret;
#endif
    case COLL_SHARM_GATHERV_ALG_XPMEM:
#if SHARM_CHECK_XPMEM_SUPPORT
        if (OPAL_UNLIKELY(SHARM_FALSE
                          == sharm_module->xpmem_runtime_check_support)) {
            OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_CALL, mca_coll_sharm_stream,
                                 "coll:sharm:%d:gatherv: (%d/%d/%s) xpmem "
                                 "runtime failed, fallback alg",
                                 SHARM_COLL(gatherv, sharm_module),
                                 ompi_comm_rank(comm), ompi_comm_size(comm),
                                 comm->c_name));
            break;
        }
        SHARM_PROFILING_TOTAL_TIME_START(sharm_module, gatherv);
        ret = sharm_gatherv_xpmem(sbuf, scount, sdtype, rbuf, rcounts, displs,
                                  rdtype, root, comm, module);
        SHARM_PROFILING_TOTAL_TIME_STOP(sharm_module, gatherv);
        return ret;
#endif
    case COLL_SHARM_GATHERV_ALG_CICO:
    default:
        break;
    }

    SHARM_PROFILING_TOTAL_TIME_START(sharm_module, gatherv);
    ret = sharm_gatherv_cico(sbuf, scount, sdtype, rbuf, rcounts, displs,
                             rdtype, root, comm, module);
    SHARM_PROFILING_TOTAL_TIME_STOP(sharm_module, gatherv);
    return ret;
}
