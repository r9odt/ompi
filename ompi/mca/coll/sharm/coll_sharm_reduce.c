/** @file */

#include "coll_sharm.h"

/*
 * External variables.
 */

extern int mca_coll_sharm_stream;
extern int mca_coll_sharm_reduce_algorithm;

/*
 * Local functions.
 */

/**
 * @brief Sharm Reduce collective operation entrypoint.
 * @param[in] sbuf send buffer.
 * @param[in,out] rbuf receive buffer.
 * @param[in] count count of elements to reduce.
 * @param[in] dtype datatype to reduce.
 * @param[in] op reduce operation which apply to data.
 * @param[in] root root process for reduce operation
 * @param[in] comm communicator for collective operation.
 * @param[in] module sharm module structure.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_reduce_intra(const void *sbuf, void *rbuf, int count,
                       struct ompi_datatype_t *dtype, struct ompi_op_t *op,
                       int root, struct ompi_communicator_t *comm,
                       mca_coll_base_module_t *module)
{
    SHARM_INIT();
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;

    int ret = OMPI_SUCCESS;

    SHARM_NEW_OP(sharm_module);
    SHARM_NEW_COLL(reduce, sharm_module);

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_CALL, mca_coll_sharm_stream,
                         "coll:sharm:%d:reduce: (%d/%d/%s) alg:%d root:%d",
                         SHARM_COLL(reduce, sharm_module), ompi_comm_rank(comm),
                         ompi_comm_size(comm), comm->c_name,
                         mca_coll_sharm_reduce_algorithm, root));

    size_t ddt_size;
    ompi_datatype_type_size(dtype, &ddt_size);
    ptrdiff_t extent;
    ompi_datatype_type_extent(dtype, &extent);

    if (ddt_size > sharm_module->shared_memory_data->mu_queue_fragment_size) {
        opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                            "coll:sharm:%d:reduce: (%d/%d/%s) "
                            "Datatype size is larger than queue fragment "
                            "(%ld > %ld) fallback to previous reduce",
                            SHARM_COLL(reduce, sharm_module),
                            ompi_comm_rank(comm), ompi_comm_size(comm),
                            comm->c_name, ddt_size,
                            shm_data->mu_queue_fragment_size);
        return sharm_module
            ->fallback_reduce(sbuf, rbuf, count, dtype, op, root, comm,
                              sharm_module->fallback_reduce_module);
    }

    switch (mca_coll_sharm_reduce_algorithm) {
    case COLL_SHARM_REDUCE_ALG_KNOMIAL:
        SHARM_PROFILING_TOTAL_TIME_START(sharm_module, reduce);
        ret = sharm_reduce_cico_knomial(sbuf, rbuf, count, dtype, op, root,
                                        comm, module);
        SHARM_PROFILING_TOTAL_TIME_STOP(sharm_module, reduce);
        return ret;
    case COLL_SHARM_REDUCE_ALG_CMA:
        if (extent > sharm_module->shared_memory_data->mu_queue_fragment_size) {
            opal_output_verbose(
                SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                "coll:sharm:%d:reduce: (%d/%d/%s) "
                "Datatype extent size is larger than queue fragment "
                "(%ld > %ld) fallback to previous reduce",
                SHARM_COLL(reduce, sharm_module), ompi_comm_rank(comm),
                ompi_comm_size(comm), comm->c_name, extent,
                shm_data->mu_queue_fragment_size);
            break;
        }
        SHARM_PROFILING_TOTAL_TIME_START(sharm_module, reduce);
        ret = sharm_reduce_cma(sbuf, rbuf, count, dtype, op, root, comm,
                               module);
        SHARM_PROFILING_TOTAL_TIME_STOP(sharm_module, reduce);
        return ret;
    case COLL_SHARM_REDUCE_ALG_XPMEM:
#if SHARM_CHECK_XPMEM_SUPPORT
        if (OPAL_UNLIKELY(SHARM_FALSE
                          == sharm_module->xpmem_runtime_check_support)) {
            opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                                "coll:sharm:%d:reduce: (%d/%d/%s) xpmem "
                                "runtime failed, fallback alg",
                                SHARM_COLL(reduce, sharm_module),
                                ompi_comm_rank(comm), ompi_comm_size(comm),
                                comm->c_name);
            break;
        }
        if (extent > sharm_module->shared_memory_data->mu_queue_fragment_size) {
            opal_output_verbose(
                SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                "coll:sharm:%d:reduce: (%d/%d/%s) "
                "Datatype extent size is larger than queue fragment "
                "(%ld > %ld) fallback to previous reduce",
                SHARM_COLL(reduce, sharm_module), ompi_comm_rank(comm),
                ompi_comm_size(comm), comm->c_name, extent,
                shm_data->mu_queue_fragment_size);
            break;
        }
        SHARM_PROFILING_TOTAL_TIME_START(sharm_module, reduce);
        ret = sharm_reduce_xpmem(sbuf, rbuf, count, dtype, op, root, comm,
                                 module);
        SHARM_PROFILING_TOTAL_TIME_STOP(sharm_module, reduce);
        return ret;
#endif
    case COLL_SHARM_REDUCE_ALG_FLAT_TREE:
    default:
        break;
    }

    SHARM_PROFILING_TOTAL_TIME_START(sharm_module, reduce);
    ret = sharm_reduce_cico_non_commutative(sbuf, rbuf, count, dtype, op, root,
                                            comm, module);
    SHARM_PROFILING_TOTAL_TIME_STOP(sharm_module, reduce);
    return ret;
}
