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
 * @brief Sharm Reduce_scatter_block collective operation entrypoint.
 * @param[in] sbuf send buffer.
 * @param[in,out] rbuf receive buffer.
 * @param[in] count count of elements to reduce.
 * @param[in] dtype datatype to reduce.
 * @param[in] op reduce operation which apply to data.
 * @param[in] comm communicator for collective operation.
 * @param[in] module sharm module structure.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_reduce_scatter_block_intra(const void *sbuf, void *rbuf, int rcount,
                                     struct ompi_datatype_t *dtype,
                                     struct ompi_op_t *op,
                                     struct ompi_communicator_t *comm,
                                     mca_coll_base_module_t *module)

{
    SHARM_INIT();
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;

    SHARM_NEW_OP(sharm_module);
    SHARM_NEW_COLL(reduce_scatter_block, sharm_module);

    int rank = ompi_comm_rank(comm);
    int size = ompi_comm_size(comm);
    int err = OMPI_SUCCESS;
    size_t count = rcount * (size_t) size;
    ptrdiff_t gap;
    ptrdiff_t span;
    char *recv_buf = NULL;
    char *recv_buf_free = NULL;

    SHARM_PROFILING_TOTAL_TIME_START(sharm_module, reduce_scatter_block);
    /* short cut the trivial case */
    if (0 == count) {
        SHARM_PROFILING_TOTAL_TIME_STOP(sharm_module, reduce_scatter_block);
        return OMPI_SUCCESS;
    }

    /* Handle MPI_IN_PLACE */
    if (MPI_IN_PLACE == sbuf) {
        sbuf = rbuf;
    }

    /* get datatype information */
    span = opal_datatype_span(&dtype->super, count, &gap);

    if (0 == rank) {
        /* temporary receive buffer.  See coll_basic_reduce.c for
           details on sizing */
        recv_buf_free = (char *) malloc(span);
        if (NULL == recv_buf_free) {
            err = OMPI_ERR_OUT_OF_RESOURCE;
            goto cleanup;
        }
        recv_buf = recv_buf_free - gap;
    }

    /* reduction */
    err = sharm_reduce_intra(sbuf, recv_buf, (int) count, dtype, op, 0, comm,
                             module);
    if (MPI_SUCCESS != err) {
        goto cleanup;
    }

    /* scatter */
    err = sharm_scatter_intra(recv_buf, rcount, dtype, rbuf, rcount, dtype, 0,
                              comm, module);

cleanup:

    if (NULL != recv_buf_free) {
        free(recv_buf_free);
    }

    SHARM_PROFILING_TOTAL_TIME_STOP(sharm_module, reduce_scatter_block);
    return err;
}
