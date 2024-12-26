/** @file */

#include "coll_sharm.h"

/*
 * External variables.
 */

extern int mca_coll_sharm_stream;
extern int mca_coll_sharm_bcast_algorithm;

/*
 * Local functions.
 */

/**
 * @brief Sharm Bcast collective operation entrypoint.
 * @param[in,out] buff send buffer.
 * @param[in] count count of elements for broadcast.
 * @param[in] datatype datatype for broadcast algorithm.
 * @param[in] root root process for broadcast algorithm.
 * @param[in] comm communicator for collective operation.
 * @param[in] module sharm module structure.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_bcast_intra(void *buff, int count, ompi_datatype_t *datatype,
                      int root, ompi_communicator_t *comm,
                      mca_coll_base_module_t *module)
{
    SHARM_INIT();
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;

    int ret = OMPI_SUCCESS;

    SHARM_NEW_OP(sharm_module);
    SHARM_NEW_COLL(bcast, sharm_module);

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_CALL, mca_coll_sharm_stream,
                         "coll:sharm:%d:bcast: (%d/%d/%s) alg:%d root:%d",
                         SHARM_COLL(bcast, sharm_module), ompi_comm_rank(comm),
                         ompi_comm_size(comm), comm->c_name,
                         mca_coll_sharm_bcast_algorithm, root));

    if (!sharm_is_single_node_mode(comm)) {
        opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                            "coll:sharm:%d:bcast: (%d/%d/%s) "
                            "Operation cannot support multiple nodes, fallback",
                            SHARM_COLL(bcast, sharm_module),
                            ompi_comm_rank(comm), ompi_comm_size(comm),
                            comm->c_name);
        return sharm_module->fallbacks
            .fallback_bcast(buff, count, datatype, root, comm,
                            sharm_module->fallbacks.fallback_bcast_module);
    }

    switch (mca_coll_sharm_bcast_algorithm) {
    case COLL_SHARM_BCAST_ALG_CMA:
#if SHARM_CHECK_CMA_SUPPORT
        SHARM_PROFILING_TOTAL_TIME_START(sharm_module, bcast);
        ret = sharm_bcast_cma(buff, count, datatype, root, comm, module);
        SHARM_PROFILING_TOTAL_TIME_STOP(sharm_module, bcast);
        return ret;
#endif
    case COLL_SHARM_BCAST_ALG_XPMEM:
#if SHARM_CHECK_XPMEM_SUPPORT
        if (OPAL_UNLIKELY(SHARM_FALSE
                          == sharm_module->xpmem_runtime_check_support)) {
            OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_CALL, mca_coll_sharm_stream,
                                 "coll:sharm:%d:bcast: (%d/%d/%s) xpmem "
                                 "runtime failed, fallback alg",
                                 SHARM_COLL(bcast, sharm_module),
                                 ompi_comm_rank(comm), ompi_comm_size(comm),
                                 comm->c_name));
            break;
        }
        SHARM_PROFILING_TOTAL_TIME_START(sharm_module, bcast);
        ret = sharm_bcast_xpmem(buff, count, datatype, root, comm, module);
        SHARM_PROFILING_TOTAL_TIME_STOP(sharm_module, bcast);
        return ret;
#endif
    case COLL_SHARM_BCAST_ALG_CICO:
    default:
        break;
    }
    SHARM_PROFILING_TOTAL_TIME_START(sharm_module, bcast);
    ret = sharm_bcast_cico(buff, count, datatype, root, comm, module);
    SHARM_PROFILING_TOTAL_TIME_STOP(sharm_module, bcast);
    return ret;
}
