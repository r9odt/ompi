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
 * @brief Sharm Scan collective operation entrypoint.
 * @param[in] sbuf send buffer.
 * @param[in,out] rbuf receive buffer.
 * @param[in] count count of elements to reduce.
 * @param[in] dtype datatype to reduce.
 * @param[in] op reduce operation which apply to data.
 * @param[in] comm communicator for collective operation.
 * @param[in] module sharm module structure.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_scan_intra(const void *sbuf, void *rbuf, int count,
                     struct ompi_datatype_t *dtype, struct ompi_op_t *op,
                     struct ompi_communicator_t *comm,
                     mca_coll_base_module_t *module)
{
    SHARM_INIT();
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;

    int ret = OMPI_SUCCESS;

    SHARM_NEW_OP(sharm_module);
    SHARM_NEW_COLL(scan, sharm_module);

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_CALL, mca_coll_sharm_stream,
                         "coll:sharm:%d:scan: (%d/%d/%s) ",
                         SHARM_COLL(scan, sharm_module), ompi_comm_rank(comm),
                         ompi_comm_size(comm), comm->c_name));

    size_t ddt_size;
    ompi_datatype_type_size(dtype, &ddt_size);

    if (ddt_size > sharm_module->shared_memory_data->mu_queue_fragment_size) {
        opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,
                            "coll:sharm:%d:scan: (%d/%d/%s) "
                            "Datatype size is larger than queue fragment "
                            "(%ld > %ld) fallback to previous scan",
                            SHARM_COLL(scan, sharm_module),
                            ompi_comm_rank(comm), ompi_comm_size(comm),
                            comm->c_name, ddt_size,
                            shm_data->mu_queue_fragment_size);
        return sharm_module->fallback_scan(sbuf, rbuf, count, dtype, op, comm,
                                           sharm_module->fallback_scan_module);
    }

    SHARM_PROFILING_TOTAL_TIME_START(sharm_module, scan);
    ret = sharm_scan_cico(sbuf, rbuf, count, dtype, op, comm, module);
    SHARM_PROFILING_TOTAL_TIME_STOP(sharm_module, scan);
    return ret;
}
