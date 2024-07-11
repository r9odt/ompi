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
 * @brief shared-memory based algorithm for Reduce_scatter
 * using CICO approach.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_reduce_scatter_cico(const void *sbuf, void *rbuf, const int *rcounts,
                              struct ompi_datatype_t *dtype,
                              struct ompi_op_t *op,
                              struct ompi_communicator_t *comm,
                              mca_coll_base_module_t *module)
{
    return OMPI_ERR_NOT_IMPLEMENTED;
}
