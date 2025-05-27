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
 * @brief shared-memory based algorithm for Exscan using XPMEM approach.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_exscan_xpmem(const void *sbuf, void *rbuf, int count,
                       struct ompi_datatype_t *dtype, struct ompi_op_t *op,
                       struct ompi_communicator_t *comm,
                       mca_coll_base_module_t *module)
{
#if !(SHARM_CHECK_XPMEM_SUPPORT)
    return OMPI_ERR_NOT_AVAILABLE;
#else
    return OMPI_ERR_NOT_SUPPORTED;
#endif
}
