/** @file */

#ifndef MCA_COLL_SHARM_PROFILING_H
#define MCA_COLL_SHARM_PROFILING_H

#include "coll_sharm_constants.h"

/*
 * Macros.
 */

// #define SHARM_PROFILING               SHARM_TRUE
// #define SHARM_PROFILING_VERBOSE_TIMES SHARM_TRUE
#ifndef SHARM_PROFILING
#    define SHARM_PROFILING SHARM_FALSE
#endif
#ifndef SHARM_PROFILING_VERBOSE_TIMES
#    define SHARM_PROFILING_VERBOSE_TIMES SHARM_FALSE
#endif

#if SHARM_PROFILING == SHARM_TRUE

/**
 * @brief Initialize counters for operation.
 * @param[in] module Pointer to coll module.
 * @param[in] operation Opreation for counters initialization.
 */
#    define SHARM_INIT_GLOBAL_PROFILING_COUNTERS(module, operation)            \
        {                                                                      \
            module->profiling_data.push_global_time_##operation = 0.0;         \
            module->profiling_data.pop_global_time_##operation = 0.0;          \
            module->profiling_data.copy_global_time_##operation = 0.0;         \
            module->profiling_data.reduce_operation_global_time_##operation    \
                = 0.0;                                                         \
            module->profiling_data.collective_exchange_global_time_##operation \
                = 0.0;                                                         \
            module->profiling_data.zcopy_barrier_global_time_##operation       \
                = 0.0;                                                         \
            module->profiling_data.total_global_time_##operation = 0.0;        \
            module->profiling_data.xpmem_attach_global_time_##operation = 0.0; \
            module->profiling_data.push_global_count_##operation = 0;          \
            module->profiling_data.pop_global_count_##operation = 0;           \
            module->profiling_data.copy_global_count_##operation = 0;          \
            module->profiling_data.reduce_operation_global_count_##operation   \
                = 0;                                                           \
            module->profiling_data.total_global_count_##operation = 0;         \
            module->profiling_data.xpmem_attach_global_count_##operation = 0;  \
            module->profiling_data                                             \
                .collective_exchange_global_count_##operation                  \
                = 0;                                                           \
            module->profiling_data.zcopy_barrier_global_count_##operation = 0; \
        }

/**
 * @brief Initialize counters for operation.
 * @param[in] operation Opreation for counters initialization.
 */
#    define SHARM_INIT_PROFILING_COUNTERS()                    \
        {                                                      \
            double push_time_##operation = 0.0;                \
            double pop_time_##operation = 0.0;                 \
            double copy_time_##operation = 0.0;                \
            double xpmem_attach_time_##operation = 0.0;        \
            double reduce_operation_time_##operation = 0.0;    \
            double total_time_##operation = 0.0;               \
            double collective_exchange_time_##operation = 0.0; \
            uint64_t push_count_##operation = 0;               \
            uint64_t pop_count_##operation = 0;                \
            uint64_t copy_count_##operation = 0;               \
            uint64_t xpmem_attach_count_##operation = 0;       \
            uint64_t reduce_operation_count_##operation = 0;   \
        }

#    define SHARM_PROFILING_TOTAL_TIME_START(module, operation)      \
        {                                                            \
            module->profiling_data.total_global_time_##operation     \
                -= MPI_Wtime();                                      \
            ++module->profiling_data.total_global_count_##operation; \
        }

#    define SHARM_PROFILING_TOTAL_TIME_STOP(module, operation)   \
        {                                                        \
            module->profiling_data.total_global_time_##operation \
                += MPI_Wtime();                                  \
        }

#    if SHARM_PROFILING_VERBOSE_TIMES == SHARM_TRUE
#        define SHARM_PROFILING_TIME_START(module, operation, type)       \
            {                                                             \
                module->profiling_data.type##_global_time_##operation     \
                    -= MPI_Wtime();                                       \
                ++module->profiling_data.type##_global_count_##operation; \
            }

#        define SHARM_PROFILING_TIME_STOP(module, operation, type)    \
            {                                                         \
                module->profiling_data.type##_global_time_##operation \
                    += MPI_Wtime();                                   \
            }
#    else
#        define SHARM_PROFILING_TIME_START(module, operation, type)
#        define SHARM_PROFILING_TIME_STOP(module, operation, type)
#    endif

#    define SHARM_PROFILING_DUMP_ALL_GLOBAL_VALUES(module)                    \
        {                                                                     \
            SHARM_PROFILING_DUMP_GLOBAL_VALUES(module, barrier);              \
            SHARM_PROFILING_DUMP_GLOBAL_VALUES(module, bcast);                \
            SHARM_PROFILING_DUMP_GLOBAL_VALUES(module, gather);               \
            SHARM_PROFILING_DUMP_GLOBAL_VALUES(module, gatherv);              \
            SHARM_PROFILING_DUMP_GLOBAL_VALUES(module, scatter);              \
            SHARM_PROFILING_DUMP_GLOBAL_VALUES(module, scatterv);             \
            SHARM_PROFILING_DUMP_GLOBAL_VALUES(module, alltoall);             \
            SHARM_PROFILING_DUMP_GLOBAL_VALUES(module, alltoallv);            \
            SHARM_PROFILING_DUMP_GLOBAL_VALUES(module, alltoallw);            \
            SHARM_PROFILING_DUMP_GLOBAL_VALUES(module, allgather);            \
            SHARM_PROFILING_DUMP_GLOBAL_VALUES(module, allgatherv);           \
            SHARM_PROFILING_DUMP_GLOBAL_VALUES(module, scan);                 \
            SHARM_PROFILING_DUMP_GLOBAL_VALUES(module, exscan);               \
            SHARM_PROFILING_DUMP_GLOBAL_VALUES(module, reduce);               \
            SHARM_PROFILING_DUMP_GLOBAL_VALUES(module, reduce_scatter);       \
            SHARM_PROFILING_DUMP_GLOBAL_VALUES(module, reduce_scatter_block); \
            SHARM_PROFILING_DUMP_GLOBAL_VALUES(module, allreduce);            \
        }

#    define SHARM_PROFILING_DUMP_GLOBAL_VALUES(module, operation)            \
        {                                                                    \
            if (SHARM_COLL(operation, module) > 0) {                         \
                opal_output(                                                 \
                    mca_coll_sharm_stream,                                   \
                    "coll:sharm:profiling:" #operation ":global (%d/%d/%s) " \
                    "total_time %.12lf total_count %d "                      \
                    "copy_time %.12lf copy_time_p %lf "                      \
                    "push_time %.12lf push_time_p %lf "                      \
                    "pop_time %.12lf pop_time_p %lf "                        \
                    "op_time %.12lf op_time_p %lf "                          \
                    "coll_ex_time %.12lf coll_ex_time_p %lf "                \
                    "xpmem_attach_time %.12lf xpmem_attach_time_p %lf "      \
                    "zcopy_barrier_time %.12lf zcopy_barrier_time_p %lf",    \
                    ompi_comm_rank(module->comm),                            \
                    ompi_comm_size(module->comm), module->comm->c_name,      \
                    module->profiling_data.total_global_time_##operation,    \
                    SHARM_COLL(operation, module),                           \
                    module->profiling_data.copy_global_time_##operation,     \
                    module->profiling_data.copy_global_time_##operation      \
                        / module->profiling_data                             \
                              .total_global_time_##operation,                \
                    module->profiling_data.push_global_time_##operation,     \
                    module->profiling_data.push_global_time_##operation      \
                        / module->profiling_data                             \
                              .total_global_time_##operation,                \
                    module->profiling_data.pop_global_time_##operation,      \
                    module->profiling_data.pop_global_time_##operation       \
                        / module->profiling_data                             \
                              .total_global_time_##operation,                \
                    module->profiling_data                                   \
                        .reduce_operation_global_time_##operation,           \
                    module->profiling_data                                   \
                            .reduce_operation_global_time_##operation        \
                        / module->profiling_data                             \
                              .total_global_time_##operation,                \
                    module->profiling_data                                   \
                        .collective_exchange_global_time_##operation,        \
                    module->profiling_data                                   \
                            .collective_exchange_global_time_##operation     \
                        / module->profiling_data                             \
                              .total_global_time_##operation,                \
                    module->profiling_data                                   \
                        .xpmem_attach_global_time_##operation,               \
                    module->profiling_data                                   \
                            .xpmem_attach_global_time_##operation            \
                        / module->profiling_data                             \
                              .total_global_time_##operation,                \
                    module->profiling_data                                   \
                        .zcopy_barrier_global_time_##operation,              \
                    module->profiling_data                                   \
                            .zcopy_barrier_global_time_##operation           \
                        / module->profiling_data                             \
                              .total_global_time_##operation);               \
            }                                                                \
        }
#else
#    define SHARM_INIT_GLOBAL_PROFILING_COUNTERS(module, operation)
#    define SHARM_INIT_PROFILING_COUNTERS()
#    define SHARM_PROFILING_TOTAL_TIME_START(module, operation)
#    define SHARM_PROFILING_TOTAL_TIME_STOP(module, operation)
#    define SHARM_PROFILING_TIME_START(module, operation, type)
#    define SHARM_PROFILING_TIME_STOP(module, operation, type)
#    define SHARM_PROFILING_DUMP_ALL_GLOBAL_VALUES(module)
#    define SHARM_PROFILING_DUMP_GLOBAL_VALUES(module, operation)
#endif

#endif /* MCA_COLL_SHARM_PROFILING_H */
