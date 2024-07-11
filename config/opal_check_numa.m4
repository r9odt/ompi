# -*- shell-script ; indent-tabs-mode:nil -*-
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

#
# special check for numa, uses macro(s) from pkg.m4
#
# OPAL_CHECK_NUMA(prefix, [action-if-found], [action-if-not-found])
# --------------------------------------------------------
# check if NUMA support can be found.  sets prefix_{CPPFLAGS,
# LDFLAGS, LIBS} as needed and runs action-if-found if there is
# support, otherwise executes action-if-not-found
AC_DEFUN([OPAL_CHECK_NUMA], [
    OPAL_VAR_SCOPE_PUSH([opal_check_numa_happy])

    AC_ARG_WITH([numa],
                [AS_HELP_STRING([--with-numa],
                                [Build with NUMA support, requires libnuma])])

    OAC_CHECK_PACKAGE([libnuma],
                      [$1],
                      [numa.h],
                      [numa],
                      [numa_node_size],
                      [opal_check_numa_happy="yes"],
                      [opal_check_numa_happy="no"])

     AS_IF([test "${opal_check_numa_happy}" = "yes"],
           [AC_DEFINE_UNQUOTED([HAVE_NUMA_H], [1], [is numa.h available])
            LIBS="${$1_LIBS} ${LIBS}"
            $2],
           [AS_IF([test -n "${with_numa}" -a "${with_numa}" != "no"],
                  [AC_MSG_ERROR([NUMA support requested but not found.  Aborting])])
            $3])

    OPAL_VAR_SCOPE_POP
])dnl
