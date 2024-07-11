/** @file */

#include "coll_sharm.h"

/*
 * Global variables.
 */
int mca_coll_sharm_priority = 0;
int mca_coll_sharm_verbose = 0;
int mca_coll_sharm_stream = -1;
int mca_coll_sharm_cacheline_size = 64;
int mca_coll_sharm_nfrags = 8;
int mca_coll_sharm_fragment_size = 8192;

char *mca_coll_sharm_segment_path = (char *) NULL;

int mca_coll_sharm_barrier_algorithm = COLL_SHARM_BARRIER_ALG_SENSE_REVERSING;

int mca_coll_sharm_bcast_algorithm = COLL_SHARM_BCAST_ALG_CICO;

int mca_coll_sharm_reduce_algorithm = COLL_SHARM_REDUCE_ALG_KNOMIAL;
int mca_coll_sharm_reduce_knomial_radix = 2;

int mca_coll_sharm_allreduce_algorithm = COLL_SHARM_ALLREDUCE_ALG_REDUCE_BCAST;

int mca_coll_sharm_gather_algorithm = COLL_SHARM_GATHER_ALG_CICO;
int mca_coll_sharm_gatherv_algorithm = COLL_SHARM_GATHERV_ALG_CICO;

int mca_coll_sharm_scatter_algorithm = COLL_SHARM_SCATTER_ALG_CICO;
int mca_coll_sharm_scatterv_algorithm = COLL_SHARM_SCATTERV_ALG_CICO;

int mca_coll_sharm_allgather_algorithm = COLL_SHARM_ALLGATHER_ALG_CICO;
int mca_coll_sharm_allgatherv_algorithm = COLL_SHARM_ALLGATHERV_ALG_CICO;

int mca_coll_sharm_alltoall_algorithm = COLL_SHARM_ALLTOALL_ALG_PAIRWISE;
int mca_coll_sharm_alltoallv_algorithm = COLL_SHARM_ALLTOALLV_ALG_PAIRWISE;
int mca_coll_sharm_alltoallw_algorithm = COLL_SHARM_ALLTOALLW_ALG_PAIRWISE;

const char *mca_coll_sharm_component_version_string
    = "Open MPI Shared Memory collective MCA component version " OMPI_VERSION;

/*
 * External functions.
 */
extern int mca_coll_sharm_init_query(bool enable_progress_threads,
                                     bool enable_mpi_threads);
extern mca_coll_base_module_t *
mca_coll_sharm_comm_query(ompi_communicator_t *comm, int *priority);

/*
 * Local functions.
 */
static int sharm_register(void);
static int sharm_open(void);
static int sharm_close(void);
static int sharm_verify_mca_variables(void);

/**
 * @brief Instantiate the public struct with all of our public information
 * and pointers to our public functions in it.
 */
mca_coll_sharm_component_t mca_coll_sharm_component = {
    /* 
     * Fill in the super.
     */
    {
        /* 
         * First, the mca_component_t struct containing meta information
         * about the component itself.
         */
        .collm_version =
            {
                MCA_COLL_BASE_VERSION_2_4_0,

                /*
                 * Component name and version.
                 */
                .mca_component_name = "sharm",
                MCA_BASE_MAKE_VERSION(component, OMPI_MAJOR_VERSION,
                                      OMPI_MINOR_VERSION, OMPI_RELEASE_VERSION),

                /*
                 * Component open and close functions.
                 */
                .mca_open_component = sharm_open,
                .mca_close_component = sharm_close,
                .mca_register_component_params = sharm_register,
            },
        .collm_data =
            {
                /*
                * The component is not checkpoint ready.
                */
                MCA_BASE_METADATA_PARAM_NONE,
            },

        /*
         * Initialization / querying functions.
         */
        .collm_init_query = mca_coll_sharm_init_query,
        .collm_comm_query = mca_coll_sharm_comm_query,
    },
    /* (default) priority */
    20,
};

static int sharm_register(void)
{
    /*
     * Priority must be greater than priority of basic component.
     */
    mca_coll_sharm_priority = 0;
    (void) mca_base_component_var_register(
        &mca_coll_sharm_component.super.collm_version, "priority",
        "Priority of the sharm coll component", MCA_BASE_VAR_TYPE_INT, NULL, 0,
        0, OPAL_INFO_LVL_9, MCA_BASE_VAR_SCOPE_ALL, &mca_coll_sharm_priority);

    (void) mca_base_component_var_register(
        &mca_coll_sharm_component.super.collm_version, "verbose",
        "Verbose level of the sharm coll component", MCA_BASE_VAR_TYPE_INT,
        NULL, 0, 0, OPAL_INFO_LVL_9, MCA_BASE_VAR_SCOPE_ALL,
        &mca_coll_sharm_verbose);

    (void) mca_base_component_var_register(
        &mca_coll_sharm_component.super.collm_version, "segment_path",
        "Prefix for shared segment path. If empty, use session directory. "
        "NOTE: Reserved 128 bytes for this string.",
        MCA_BASE_VAR_TYPE_STRING, NULL, 0, 0, OPAL_INFO_LVL_9,
        MCA_BASE_VAR_SCOPE_ALL, &mca_coll_sharm_segment_path);

    (void) mca_base_component_var_register(&mca_coll_sharm_component.super
                                                .collm_version,
                                           "cacheline_size",
                                           "Override cacheline size",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_ALL,
                                           &mca_coll_sharm_cacheline_size);

    mca_coll_sharm_nfrags = 8;
    (void) mca_base_component_var_register(&mca_coll_sharm_component.super
                                                .collm_version,
                                           "nsegs", "Total number of segments",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_ALL,
                                           &mca_coll_sharm_nfrags);

    mca_coll_sharm_fragment_size = 2 * opal_getpagesize();
    (void) mca_base_component_var_register(
        &mca_coll_sharm_component.super.collm_version, "fragment_size",
        "fragment size (bytes)", MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
        OPAL_INFO_LVL_9, MCA_BASE_VAR_SCOPE_ALL, &mca_coll_sharm_fragment_size);

    (void) mca_base_component_var_register(
        &mca_coll_sharm_component.super.collm_version, "cacheline_size",
        "Verbose level of the sharm coll component", MCA_BASE_VAR_TYPE_INT,
        NULL, 0, 0, OPAL_INFO_LVL_9, MCA_BASE_VAR_SCOPE_ALL,
        &mca_coll_sharm_cacheline_size);

    /*
     * Algorithm parameters
     */

    (void) mca_base_component_var_register(
        &mca_coll_sharm_component.super.collm_version, "barrier_algorithm",
        "Algorithm for barrier operation: 1 - sense-reversing (default), 2 - "
        "cico",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_9,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_sharm_barrier_algorithm);

    (void) mca_base_component_var_register(
        &mca_coll_sharm_component.super.collm_version, "bcast_algorithm",
        "Algorithm for bcast operation: 1 - cico (default), 11 - cma, 21 - "
        "xpmem",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_9,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_sharm_bcast_algorithm);

    (void) mca_base_component_var_register(
        &mca_coll_sharm_component.super.collm_version, "reduce_algorithm",
        "Algorithm for reduce operation: 1 - flat, 2 - knomial (default), 21 - "
        "xpmem",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_9,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_sharm_reduce_algorithm);

    (void) mca_base_component_var_register(
        &mca_coll_sharm_component.super.collm_version, "reduce_knomial_radix",
        "Radix for knomial reduce, default - 2", MCA_BASE_VAR_TYPE_INT, NULL, 0,
        0, OPAL_INFO_LVL_9, MCA_BASE_VAR_SCOPE_READONLY,
        &mca_coll_sharm_reduce_knomial_radix);

    (void) mca_base_component_var_register(
        &mca_coll_sharm_component.super.collm_version, "allreduce_algorithm",
        "Algorithm for reduce operation: 1 - reduce+bcast (default), 2 - "
        "allreduce native reduce+bcast, 3 - native flat",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_9,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_sharm_allreduce_algorithm);

    (void) mca_base_component_var_register(
        &mca_coll_sharm_component.super.collm_version, "scatter_algorithm",
        "Algorithm for scatter operation: 1 - cico (default), 11 - cma, 21 - "
        "xpmem",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_9,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_sharm_scatter_algorithm);

    (void) mca_base_component_var_register(
        &mca_coll_sharm_component.super.collm_version, "scatterv_algorithm",
        "Algorithm for scatterv operation: 1 - cico (default), 11 - cma, 21 - "
        "xpmem",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_9,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_sharm_scatterv_algorithm);

    (void) mca_base_component_var_register(
        &mca_coll_sharm_component.super.collm_version, "gather_algorithm",
        "Algorithm for gather operation: 1 - cico (default), 11 - cma, 21 - "
        "xpmem",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_9,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_sharm_gather_algorithm);

    (void) mca_base_component_var_register(
        &mca_coll_sharm_component.super.collm_version, "gatherv_algorithm",
        "Algorithm for gatherv operation: 1 - cico (default), 11 - cma, 21 - "
        "xpmem",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_9,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_sharm_gatherv_algorithm);

    (void) mca_base_component_var_register(
        &mca_coll_sharm_component.super.collm_version, "allgather_algorithm",
        "Algorithm for allgather operation: 1 - cico (default), 11 - cma, 21 - "
        "xpmem",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_9,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_sharm_allgather_algorithm);

    (void) mca_base_component_var_register(
        &mca_coll_sharm_component.super.collm_version, "allgatherv_algorithm",
        "Algorithm for allgatherv operation: 1 - cico (default), 11 - cma, 21 "
        "- xpmem",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_9,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_sharm_allgatherv_algorithm);

    (void) mca_base_component_var_register(
        &mca_coll_sharm_component.super.collm_version, "alltoall_algorithm",
        "Algorithm for alltoall operation: 1 - pairwise (default), 11 - cma, "
        "21 - xpmem",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_9,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_sharm_alltoall_algorithm);

    (void) mca_base_component_var_register(
        &mca_coll_sharm_component.super.collm_version, "alltoallv_algorithm",
        "Algorithm for alltoallv operation: 1 - pairwise (default), 11 - cma, "
        "21 - xpmem",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_9,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_sharm_alltoallv_algorithm);

    (void) mca_base_component_var_register(
        &mca_coll_sharm_component.super.collm_version, "alltoallw_algorithm",
        "Algorithm for alltoallw operation: 1 - pairwise (default), 11 - cma, "
        "21 - xpmem",
        MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_9,
        MCA_BASE_VAR_SCOPE_READONLY, &mca_coll_sharm_alltoallw_algorithm);
    return sharm_verify_mca_variables();
}

static int sharm_open(void)
{
    mca_coll_sharm_stream = opal_output_open(NULL);
    opal_output_set_verbosity(mca_coll_sharm_stream, mca_coll_sharm_verbose);
    opal_output_verbose(SHARM_LOG_INFO, mca_coll_sharm_stream,
                        "coll:sharm:component_open:");
    return OMPI_SUCCESS;
}

static int sharm_close(void)
{
    return OMPI_SUCCESS;
}

static int sharm_verify_mca_variables(void)
{
    if (mca_coll_sharm_cacheline_size <= 0) {
        mca_coll_sharm_cacheline_size = 64;
    }

    if (mca_coll_sharm_nfrags <= 0) {
        mca_coll_sharm_nfrags = 8;
    }

    if (mca_coll_sharm_fragment_size <= 0) {
        mca_coll_sharm_nfrags = 4096;
    }

    if (mca_coll_sharm_fragment_size < mca_coll_sharm_cacheline_size) {
        mca_coll_sharm_fragment_size = mca_coll_sharm_cacheline_size;
        opal_output_verbose(
            SHARM_LOG_PARAMETERS, mca_coll_sharm_stream,
            "coll:sharm: "
            "Fragment size is lesser than cacheline - up to 64");
    }

    if (mca_coll_sharm_barrier_algorithm <= 0
        || COLL_SHARM_BARRIER_ALG_CICO < mca_coll_sharm_barrier_algorithm) {
        opal_output_verbose(
            SHARM_LOG_PARAMETERS, mca_coll_sharm_stream,
            "coll:sharm: "
            "Barrier algorithm is out of range - set to sense-reversing");
        mca_coll_sharm_barrier_algorithm
            = COLL_SHARM_BARRIER_ALG_SENSE_REVERSING;
    }

    if ((mca_coll_sharm_bcast_algorithm <= 0
         || COLL_SHARM_BCAST_ALG_CICO < mca_coll_sharm_bcast_algorithm)
        && (mca_coll_sharm_bcast_algorithm < 10
            || COLL_SHARM_BCAST_ALG_CMA < mca_coll_sharm_bcast_algorithm)
        && (mca_coll_sharm_bcast_algorithm < 20
            || COLL_SHARM_BCAST_ALG_XPMEM < mca_coll_sharm_bcast_algorithm)) {
        opal_output_verbose(SHARM_LOG_PARAMETERS, mca_coll_sharm_stream,
                            "coll:sharm: "
                            "Bcast algorithm is out of range - set to cico");
        mca_coll_sharm_bcast_algorithm = COLL_SHARM_BCAST_ALG_CICO;
    }

    if ((mca_coll_sharm_scatter_algorithm <= 0
         || COLL_SHARM_SCATTER_ALG_CICO < mca_coll_sharm_scatter_algorithm)
        && (mca_coll_sharm_scatter_algorithm < 10
            || COLL_SHARM_SCATTER_ALG_CMA < mca_coll_sharm_scatter_algorithm)
        && (mca_coll_sharm_scatter_algorithm < 20
            || COLL_SHARM_SCATTER_ALG_XPMEM
                   < mca_coll_sharm_scatter_algorithm)) {
        opal_output_verbose(
            SHARM_LOG_PARAMETERS, mca_coll_sharm_stream,
            "coll:sharm: "
            "Scatter algorithm is out of range - set to default pairwize");
        mca_coll_sharm_scatter_algorithm = COLL_SHARM_SCATTER_ALG_CICO;
    }

    if ((mca_coll_sharm_scatterv_algorithm <= 0
         || COLL_SHARM_SCATTERV_ALG_CICO < mca_coll_sharm_scatterv_algorithm)
        && (mca_coll_sharm_scatterv_algorithm < 10
            || COLL_SHARM_SCATTERV_ALG_CMA < mca_coll_sharm_scatterv_algorithm)
        && (mca_coll_sharm_scatterv_algorithm < 20
            || COLL_SHARM_SCATTERV_ALG_XPMEM
                   < mca_coll_sharm_scatterv_algorithm)) {
        opal_output_verbose(
            SHARM_LOG_PARAMETERS, mca_coll_sharm_stream,
            "coll:sharm: "
            "Scatterv algorithm is out of range - set to default pairwize");
        mca_coll_sharm_scatterv_algorithm = COLL_SHARM_SCATTERV_ALG_CICO;
    }

    if ((mca_coll_sharm_gather_algorithm <= 0
         || COLL_SHARM_GATHER_ALG_CICO < mca_coll_sharm_gather_algorithm)
        && (mca_coll_sharm_gather_algorithm < 10
            || COLL_SHARM_GATHER_ALG_CMA < mca_coll_sharm_gather_algorithm)
        && (mca_coll_sharm_gather_algorithm < 20
            || COLL_SHARM_GATHER_ALG_XPMEM < mca_coll_sharm_gather_algorithm)) {
        opal_output_verbose(
            SHARM_LOG_PARAMETERS, mca_coll_sharm_stream,
            "coll:sharm: "
            "Gather algorithm is out of range - set to default pairwize");
        mca_coll_sharm_gather_algorithm = COLL_SHARM_GATHER_ALG_CICO;
    }

    if ((mca_coll_sharm_gatherv_algorithm <= 0
         || COLL_SHARM_GATHERV_ALG_CICO < mca_coll_sharm_gatherv_algorithm)
        && (mca_coll_sharm_gatherv_algorithm < 10
            || COLL_SHARM_GATHERV_ALG_CMA < mca_coll_sharm_gatherv_algorithm)
        && (mca_coll_sharm_gatherv_algorithm < 20
            || COLL_SHARM_GATHERV_ALG_XPMEM
                   < mca_coll_sharm_gatherv_algorithm)) {
        opal_output_verbose(
            SHARM_LOG_PARAMETERS, mca_coll_sharm_stream,
            "coll:sharm: "
            "Gatherv algorithm is out of range - set to default pairwize");
        mca_coll_sharm_gatherv_algorithm = COLL_SHARM_GATHERV_ALG_CICO;
    }

    if ((mca_coll_sharm_reduce_algorithm <= 0
         || COLL_SHARM_REDUCE_ALG_KNOMIAL < mca_coll_sharm_reduce_algorithm)
        && (mca_coll_sharm_reduce_algorithm < 10
            || COLL_SHARM_REDUCE_ALG_CMA < mca_coll_sharm_reduce_algorithm)
        && (mca_coll_sharm_reduce_algorithm < 20
            || COLL_SHARM_REDUCE_ALG_XPMEM < mca_coll_sharm_reduce_algorithm)) {
        opal_output_verbose(
            SHARM_LOG_PARAMETERS, mca_coll_sharm_stream,
            "coll:sharm: "
            "Reduce algorithm is out of range - set to knomial");
        mca_coll_sharm_reduce_algorithm = COLL_SHARM_REDUCE_ALG_KNOMIAL;
    }

    if (mca_coll_sharm_allreduce_algorithm <= 0
        || COLL_SHARM_ALLREDUCE_ALG_NATIVE_REDUCE_BROADCAST
               < mca_coll_sharm_allreduce_algorithm) {
        opal_output_verbose(SHARM_LOG_PARAMETERS, mca_coll_sharm_stream,
                            "coll:sharm: "
                            "Allreduce algorithm is out of range - set to "
                            "default reduce+bcast");
        mca_coll_sharm_allreduce_algorithm
            = COLL_SHARM_ALLREDUCE_ALG_REDUCE_BCAST;
    }

    if ((mca_coll_sharm_allgather_algorithm <= 0
         || COLL_SHARM_ALLGATHER_ALG_CICO < mca_coll_sharm_allgather_algorithm)
        && (mca_coll_sharm_allgather_algorithm < 10
            || COLL_SHARM_ALLGATHER_ALG_CMA
                   < mca_coll_sharm_allgather_algorithm)
        && (mca_coll_sharm_allgather_algorithm < 20
            || COLL_SHARM_ALLGATHER_ALG_XPMEM
                   < mca_coll_sharm_allgather_algorithm)) {
        opal_output_verbose(
            SHARM_LOG_PARAMETERS, mca_coll_sharm_stream,
            "coll:sharm: "
            "Allgather algorithm is out of range - set to default pairwize");
        mca_coll_sharm_allgather_algorithm = COLL_SHARM_ALLGATHER_ALG_CICO;
    }

    if ((mca_coll_sharm_allgatherv_algorithm <= 0
         || COLL_SHARM_ALLGATHERV_ALG_CICO
                < mca_coll_sharm_allgatherv_algorithm)
        && (mca_coll_sharm_allgatherv_algorithm < 10
            || COLL_SHARM_ALLGATHERV_ALG_CMA
                   < mca_coll_sharm_allgatherv_algorithm)
        && (mca_coll_sharm_allgatherv_algorithm < 20
            || COLL_SHARM_ALLGATHERV_ALG_XPMEM
                   < mca_coll_sharm_allgatherv_algorithm)) {
        opal_output_verbose(
            SHARM_LOG_PARAMETERS, mca_coll_sharm_stream,
            "coll:sharm: "
            "Allgatherv algorithm is out of range - set to default pairwize");
        mca_coll_sharm_allgatherv_algorithm = COLL_SHARM_ALLGATHERV_ALG_CICO;
    }

    if ((mca_coll_sharm_alltoall_algorithm <= 0
         || COLL_SHARM_ALLTOALL_ALG_PAIRWISE
                < mca_coll_sharm_alltoall_algorithm)
        && (mca_coll_sharm_alltoall_algorithm < 10
            || COLL_SHARM_ALLTOALL_ALG_CMA < mca_coll_sharm_alltoall_algorithm)
        && (mca_coll_sharm_alltoall_algorithm < 20
            || COLL_SHARM_ALLTOALL_ALG_XPMEM
                   < mca_coll_sharm_alltoall_algorithm)) {
        opal_output_verbose(
            SHARM_LOG_PARAMETERS, mca_coll_sharm_stream,
            "coll:sharm: "
            "Alltoall algorithm is out of range - set to default pairwize");
        mca_coll_sharm_alltoall_algorithm = COLL_SHARM_ALLTOALL_ALG_PAIRWISE;
    }

    if ((mca_coll_sharm_alltoallv_algorithm <= 0
         || COLL_SHARM_ALLTOALLV_ALG_PAIRWISE
                < mca_coll_sharm_alltoallv_algorithm)
        && (mca_coll_sharm_alltoallv_algorithm < 10
            || COLL_SHARM_ALLTOALLV_ALG_CMA
                   < mca_coll_sharm_alltoallv_algorithm)
        && (mca_coll_sharm_alltoallv_algorithm < 20
            || COLL_SHARM_ALLTOALLV_ALG_XPMEM
                   < mca_coll_sharm_alltoallv_algorithm)) {
        opal_output_verbose(
            SHARM_LOG_PARAMETERS, mca_coll_sharm_stream,
            "coll:sharm: "
            "Alltoallv algorithm is out of range - set to default pairwize");
        mca_coll_sharm_alltoallv_algorithm = COLL_SHARM_ALLTOALLV_ALG_PAIRWISE;
    }

    if ((mca_coll_sharm_alltoallw_algorithm <= 0
         || COLL_SHARM_ALLTOALLW_ALG_PAIRWISE
                < mca_coll_sharm_alltoallw_algorithm)
        && (mca_coll_sharm_alltoallw_algorithm < 10
            || COLL_SHARM_ALLTOALLW_ALG_CMA
                   < mca_coll_sharm_alltoallw_algorithm)
        && (mca_coll_sharm_alltoallw_algorithm < 20
            || COLL_SHARM_ALLTOALLW_ALG_XPMEM
                   < mca_coll_sharm_alltoallw_algorithm)) {
        opal_output_verbose(
            SHARM_LOG_PARAMETERS, mca_coll_sharm_stream,
            "coll:sharm: "
            "Alltoallw algorithm is out of range - set to default pairwize");
        mca_coll_sharm_alltoallv_algorithm = COLL_SHARM_ALLTOALLW_ALG_PAIRWISE;
    }

    return OMPI_SUCCESS;
}
