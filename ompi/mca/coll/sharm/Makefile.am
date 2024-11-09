#
# Copyright (c) 2022-2023 Siberian State University of Telecommunications
#                         and Information Science. All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

AM_CPPFLAGS = $(coll_sharm_CPPFLAGS)

sources = \
        coll_sharm.h \
        coll_sharm_func.h \
        coll_sharm_datatype.h \
        coll_sharm_algorithm_types.h \
        coll_sharm_profiling.h \
        coll_sharm_constants.h \
        coll_sharm_util.h \
        coll_sharm_util.c \
        coll_sharm_component.c \
        coll_sharm_segment.c \
        coll_sharm_queue.c \
        coll_sharm_queue_push.c \
        coll_sharm_queue_pop.c \
        coll_sharm_module.c \
        coll_sharm_barrier.c \
        coll_sharm_bcast.c \
        coll_sharm_bcast_cico.c \
        coll_sharm_bcast_cma.c \
        coll_sharm_bcast_xpmem.c \
        coll_sharm_scatter.c \
        coll_sharm_scatter_cico.c \
        coll_sharm_scatter_cma.c \
        coll_sharm_scatter_xpmem.c \
        coll_sharm_scatterv.c \
        coll_sharm_scatterv_cico.c \
        coll_sharm_scatterv_cma.c \
        coll_sharm_scatterv_xpmem.c \
        coll_sharm_gather.c \
        coll_sharm_gather_cico.c \
        coll_sharm_gather_cma.c \
        coll_sharm_gather_xpmem.c \
        coll_sharm_gatherv.c \
        coll_sharm_gatherv_cico.c \
        coll_sharm_gatherv_cma.c \
        coll_sharm_gatherv_xpmem.c \
        coll_sharm_allgather.c \
        coll_sharm_allgather_cico.c \
        coll_sharm_allgather_cma.c \
        coll_sharm_allgather_xpmem.c \
        coll_sharm_allgatherv.c \
        coll_sharm_allgatherv_cico.c \
        coll_sharm_allgatherv_cma.c \
        coll_sharm_allgatherv_xpmem.c \
        coll_sharm_reduce.c \
        coll_sharm_reduce_cico.c \
        coll_sharm_reduce_cico_knomial.c \
        coll_sharm_reduce_cma.c \
        coll_sharm_reduce_xpmem.c \
        coll_sharm_reduce_scatter.c \
        coll_sharm_reduce_scatter_cico.c \
        coll_sharm_reduce_scatter_block.c \
        coll_sharm_reduce_scatter_block_cico.c \
        coll_sharm_scan.c \
        coll_sharm_scan_cico.c \
        coll_sharm_scan_xpmem.c \
        coll_sharm_exscan.c \
        coll_sharm_exscan_cico.c \
        coll_sharm_alltoall.c \
        coll_sharm_alltoall_cico.c \
        coll_sharm_alltoall_cma.c \
        coll_sharm_alltoall_xpmem.c \
        coll_sharm_alltoallw.c \
        coll_sharm_alltoallw_cico.c \
        coll_sharm_alltoallw_cma.c \
        coll_sharm_alltoallw_xpmem.c \
        coll_sharm_alltoallv.c \
        coll_sharm_alltoallv_cico.c \
        coll_sharm_alltoallv_cma.c \
        coll_sharm_alltoallv_xpmem.c \
        coll_sharm_allreduce.c \
        coll_sharm_allreduce_cico_rbcast.c \
        coll_sharm_allreduce_cico.c

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_ompi_coll_sharm_DSO
component_noinst =
component_install = mca_coll_sharm.la
else
component_noinst = libmca_coll_sharm.la
component_install =
endif

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_coll_sharm_la_SOURCES = $(sources)
mca_coll_sharm_la_LDFLAGS = -module -avoid-version $(coll_sharm_LDFLAGS)
mca_coll_sharm_la_LIBADD = $(top_builddir)/ompi/lib@OMPI_LIBMPI_NAME@.la \
                           $(coll_sharm_LIBS)

noinst_LTLIBRARIES = $(component_noinst)
libmca_coll_sharm_la_SOURCES = $(sources)
libmca_coll_sharm_la_LDFLAGS = -module -avoid-version $(coll_sharm_LDFLAGS)
