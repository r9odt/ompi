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
 * @brief Sharm Reduce_scatter collective operation entrypoint.
 * @param[in] sbuf send buffer.
 * @param[in,out] rbuf receive buffer.
 * @param[in] count count of elements to reduce.
 * @param[in] dtype datatype to reduce.
 * @param[in] op reduce operation which apply to data.
 * @param[in] comm communicator for collective operation.
 * @param[in] module sharm module structure.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_reduce_scatter_intra(const void *sbuf, void *rbuf, const int *rcounts,
                               struct ompi_datatype_t *dtype,
                               struct ompi_op_t *op,
                               struct ompi_communicator_t *comm,
                               mca_coll_base_module_t *module)
{
    SHARM_INIT();
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;

    SHARM_NEW_OP(sharm_module);
    SHARM_NEW_COLL(reduce_scatter, sharm_module);

    int comm_rank = ompi_comm_rank(comm);
    int comm_size = ompi_comm_size(comm);

    OPAL_OUTPUT_VERBOSE(
        (SHARM_LOG_FUNCTION_CALL, mca_coll_sharm_stream,
         "coll:sharm:%d:reduce_scatter: (%d/%d/%s) sbuf:%p rbuf:%p "
         "rcounts:%p dtype:%p op:%p",
         SHARM_COLL(reduce_scatter, sharm_module), ompi_comm_rank(comm),
         ompi_comm_size(comm), comm->c_name, sbuf, rbuf, (void *) rcounts,
         (void *) dtype, (void *) op));

    // It.s from ompi coll/base
    int ret = OMPI_SUCCESS;
    int reduce_root = SHARM_COLL(reduce_scatter, sharm_module) % comm_size;
    int count = rcounts[0];
    char *tmprbuf = (char *) rbuf;
    char *tmprbuf_free = NULL;

    SHARM_PROFILING_TOTAL_TIME_START(sharm_module, reduce_scatter);
    int *displs = (int *) malloc(comm_size * sizeof(int));
    displs[0] = 0;
    for (int i = 1; i < comm_size; i++) {
        displs[i] = displs[i - 1] + rcounts[i - 1];
        count += rcounts[i];
    }
    if (MPI_IN_PLACE == sbuf) {
        if (reduce_root == comm_rank) {
            ret = sharm_reduce_intra(MPI_IN_PLACE, tmprbuf, count, dtype, op,
                                     reduce_root, comm, module);
        } else {
            ret = sharm_reduce_intra(tmprbuf, NULL, count, dtype, op,
                                     reduce_root, comm, module);
        }
    } else {
        if (reduce_root == comm_rank) {
            /* We must allocate temporary receive buffer on root to ensure that
               rbuf is big enough */
            ptrdiff_t dsize, gap = 0;
            dsize = opal_datatype_span(&dtype->super, count, &gap);

            tmprbuf_free = (char *) malloc(dsize);
            tmprbuf = tmprbuf_free - gap;
        }
        ret = sharm_reduce_intra(sbuf, tmprbuf, count, dtype, op, reduce_root,
                                 comm, module);
    }

    if (MPI_SUCCESS != ret) {
        if (NULL != tmprbuf_free) {
            free(tmprbuf_free);
        }

        SHARM_PROFILING_TOTAL_TIME_STOP(sharm_module, reduce_scatter);
        return ret;
    }

    if (MPI_IN_PLACE == sbuf && reduce_root == comm_rank) {
        ret = sharm_scatterv_intra(tmprbuf, rcounts, displs, dtype,
                                   MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                                   reduce_root, comm,
                                   comm->c_coll->coll_scatterv_module);
    } else {
        ret = sharm_scatterv_intra(tmprbuf, rcounts, displs, dtype, rbuf,
                                   rcounts[comm_rank], dtype, reduce_root,
                                   comm, comm->c_coll->coll_scatterv_module);
    }

    free(displs);

    if (NULL != tmprbuf_free) {
        free(tmprbuf_free);
    }

    SHARM_PROFILING_TOTAL_TIME_STOP(sharm_module, reduce_scatter);
    return ret;
}
