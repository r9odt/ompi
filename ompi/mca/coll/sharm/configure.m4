# -*- autoconf -*-
#
# Copyright (c) 2024 Computer Systems Department, SibSUTIS or its affiliates.  All Rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

AC_DEFUN([MCA_ompi_coll_sharm_CONFIG],[
    AC_CONFIG_FILES([ompi/mca/coll/sharm/Makefile])

    OPAL_CHECK_NUMA([coll_sharm],
                     [coll_sharm_numa="yes"],
                     [coll_sharm_numa="no"])

     AS_IF([test "${coll_sharm_numa}" = "yes"],
           [AC_DEFINE_UNQUOTED([SHARM_NUMA_SUPPORT], [1], [is numa.h available])],
           [AC_DEFINE_UNQUOTED([SHARM_NUMA_SUPPORT], [0], [is numa.h available])])


    OPAL_CHECK_CMA([coll_sharm],
                     [coll_sharm_cma="yes"],
                     [coll_sharm_cma="no"])
     AS_IF([test "${coll_sharm_cma}" = "yes"],
           [AC_DEFINE_UNQUOTED([SHARM_CMA_SUPPORT], [1], [is cma available])],
           [AC_DEFINE_UNQUOTED([SHARM_CMA_SUPPORT], [0], [is cma available])])


    OPAL_CHECK_KNEM([coll_sharm],
                     [coll_sharm_knem="yes"],
                     [coll_sharm_knem="no"])
     AS_IF([test "${coll_sharm_knem}" = "yes"],
           [AC_DEFINE_UNQUOTED([SHARM_KNEM_SUPPORT], [1], [is knem available])],
           [AC_DEFINE_UNQUOTED([SHARM_KNEM_SUPPORT], [0], [is knem available])])


    OPAL_CHECK_XPMEM([coll_sharm],
                     [coll_sharm_xpmem="yes"],
                     [coll_sharm_xpmem="no"])
     AS_IF([test "${coll_sharm_xpmem}" = "yes"],
           [AC_DEFINE_UNQUOTED([SHARM_XPMEM_SUPPORT], [1], [is xpmem available])],
           [AC_DEFINE_UNQUOTED([SHARM_XPMEM_SUPPORT], [0], [is xpmem available])])

    AC_SUBST([coll_sharm_CPPFLAGS])
    AC_SUBST([coll_sharm_LDFLAGS])
    AC_SUBST([coll_sharm_LIBS])
])dnl
