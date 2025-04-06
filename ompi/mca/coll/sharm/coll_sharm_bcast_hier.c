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
 * @brief Sharm Bcast collective operation hier selection.
 * @param[in,out] buff send buffer.
 * @param[in] count count of elements for broadcast.
 * @param[in] datatype datatype for broadcast algorithm.
 * @param[in] root root process for broadcast algorithm.
 * @param[in] comm communicator for collective operation.
 * @param[in] module sharm module structure.
 * @return OMPI_SUCCESS or error code.
 */
int sharm_bcast_hier_intra(void *buff, int count, ompi_datatype_t *datatype,
                           int root, ompi_communicator_t *comm,
                           mca_coll_base_module_t *module)
{
    mca_coll_sharm_module_t *sharm_module = (mca_coll_sharm_module_t *) module;

    int ret = OMPI_SUCCESS;
    // ret = sharm_bcast_intra(buff, count, datatype, root,
    //                         sharm_module->shared_comm,
    //                         sharm_module->shared_comm->c_coll
    //                             ->coll_bcast_module);
    return sharm_module->fallbacks
        .fallback_bcast(buff, count, datatype, root, comm,
                        sharm_module->fallbacks.fallback_bcast_module);
}
