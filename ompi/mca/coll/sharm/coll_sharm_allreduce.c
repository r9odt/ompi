/** @file */

#include "coll_sharm.h"

/*
 * External variables.
 */

extern int mca_coll_sharm_stream;
extern int mca_coll_sharm_allreduce_algorithm;

/*
 * Local functions.
 */

#define FALLBACK_CHECK                                                         \
    {                                                                          \
        size_t ddt_size;                                                       \
        ompi_datatype_type_size(dtype, &ddt_size);                             \
        if (ddt_size                                                           \
            > sharm_module->shared_memory_data->mu_queue_fragment_size) {      \
            opal_output_verbose(SHARM_LOG_ALWAYS, mca_coll_sharm_stream,       \
                                "coll:sharm:%d:allreduce_cico: (%d/%d/%s) "    \
                                "Datatype size is larger than queue fragment " \
                                "(%lu > %lu) fallback to previous reduce",     \
                                SHARM_OP(sharm_module), ompi_comm_rank(comm),  \
                                ompi_comm_size(comm), comm->c_name, ddt_size,  \
                                shm_data->mu_queue_fragment_size);             \
            return sharm_module                                                \
                ->fallback_allreduce(sbuf, rbuf, count, dtype, op, comm,       \
                                     sharm_module->fallback_allreduce_module); \
        }                                                                      \
    }

/**
 * @brief Sharm Allreduce collective operation entrypoint.
 * @param[in] sbuf send buffer.
 * @param[in,out] rbuf receive buffer.
 * @param[in] count count of elements to reduce
 * @param[in] dtype datatype to reduce.
 * @param[in] op reduce operation which apply to data.
 * @param[in] comm communicator for collective operation.
 * @param[in] module sharm module structure.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_allreduce_intra(const void *sbuf, void *rbuf, int count,
                          struct ompi_datatype_t *dtype, struct ompi_op_t *op,
                          struct ompi_communicator_t *comm,
                          mca_coll_base_module_t *module)
{
    SHARM_INIT();
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;
    sharm_coll_data_t *shm_data = sharm_module->shared_memory_data;
    int comm_rank = ompi_comm_rank(comm);
    int comm_size = ompi_comm_size(comm);

    int ret = OMPI_SUCCESS;

    SHARM_NEW_OP(sharm_module);
    SHARM_NEW_COLL(allreduce, sharm_module);

    OPAL_OUTPUT_VERBOSE((SHARM_LOG_FUNCTION_CALL, mca_coll_sharm_stream,
                         "coll:sharm:%d:allreduce: (%d/%d/%s) alg:%d",
                         SHARM_COLL(allreduce, sharm_module), comm_rank,
                         comm_size, comm->c_name,
                         mca_coll_sharm_allreduce_algorithm));

    switch (mca_coll_sharm_allreduce_algorithm) {
    case COLL_SHARM_ALLREDUCE_ALG_FLAT_TREE:
        FALLBACK_CHECK;
        SHARM_PROFILING_TOTAL_TIME_START(sharm_module, allreduce);
        ret = sharm_allreduce_cico_non_commutative(sbuf, rbuf, count, dtype, op,
                                                   comm, module);

        SHARM_PROFILING_TOTAL_TIME_STOP(sharm_module, reduce);
        return ret;
    case COLL_SHARM_ALLREDUCE_ALG_NATIVE_REDUCE_BROADCAST:
        FALLBACK_CHECK;
        SHARM_PROFILING_TOTAL_TIME_START(sharm_module, allreduce);
        ret = sharm_allreduce_cico_reduce_broadcast(sbuf, rbuf, count, dtype,
                                                    op, comm, module);

        SHARM_PROFILING_TOTAL_TIME_STOP(sharm_module, allreduce);
        return ret;
    case COLL_SHARM_ALLREDUCE_ALG_REDUCE_BCAST:
    default:
        break;
    }

    int reduce_root = SHARM_COLL(allreduce, sharm_module) % comm_size;
    SHARM_PROFILING_TOTAL_TIME_START(sharm_module, allreduce);
    if (MPI_IN_PLACE == sbuf) {
        if (reduce_root == comm_rank) {
            ret = sharm_reduce_intra(sbuf, rbuf, count, dtype, op, reduce_root,
                                     comm, module);
        } else {
            ret = sharm_reduce_intra(rbuf, NULL, count, dtype, op, reduce_root,
                                     comm, module);
        }
    } else {
        ret = sharm_reduce_intra(sbuf, rbuf, count, dtype, op, reduce_root,
                                 comm, module);
    }

    if (OMPI_SUCCESS == ret) {
        ret = sharm_bcast_intra(rbuf, count, dtype, reduce_root, comm, module);
    }
    SHARM_PROFILING_TOTAL_TIME_STOP(sharm_module, allreduce);
    return ret;
}
