/*
 * $HEADER$
 */

#include "lam_config.h"

#include <stdio.h>

#include "mpi.h"
#include "mpi/f77/bindings.h"
#include "group/group.h"

#if LAM_HAVE_WEAK_SYMBOLS && LAM_PROFILE_LAYER
#pragma weak PMPI_GROUP_INCL = mpi_group_incl_f
#pragma weak pmpi_group_incl = mpi_group_incl_f
#pragma weak pmpi_group_incl_ = mpi_group_incl_f
#pragma weak pmpi_group_incl__ = mpi_group_incl_f
#elif LAM_PROFILE_LAYER
LAM_GENERATE_F77_BINDINGS (PMPI_GROUP_INCL,
                           pmpi_group_incl,
                           pmpi_group_incl_,
                           pmpi_group_incl__,
                           pmpi_group_incl_f,
                           (MPI_Fint *group, MPI_Fint *n, MPI_Fint *ranks, MPI_Fint *newgroup, MPI_Fint *ierr),
                           (group, n, ranks, newgroup, ierr) )
#endif

#if LAM_HAVE_WEAK_SYMBOLS
#pragma weak MPI_GROUP_INCL = mpi_group_incl_f
#pragma weak mpi_group_incl = mpi_group_incl_f
#pragma weak mpi_group_incl_ = mpi_group_incl_f
#pragma weak mpi_group_incl__ = mpi_group_incl_f
#endif

#if ! LAM_HAVE_WEAK_SYMBOLS && ! LAM_PROFILE_LAYER
LAM_GENERATE_F77_BINDINGS (MPI_GROUP_INCL,
                           mpi_group_incl,
                           mpi_group_incl_,
                           mpi_group_incl__,
                           mpi_group_incl_f,
                           (MPI_Fint *group, MPI_Fint *n, MPI_Fint *ranks, MPI_Fint *newgroup, MPI_Fint *ierr),
                           (group, n, ranks, newgroup, ierr) )
#endif


#if LAM_PROFILE_LAYER && ! LAM_HAVE_WEAK_SYMBOLS
#include "mpi/c/profile/defines.h"
#endif

void mpi_group_incl_f(MPI_Fint *group, MPI_Fint *n, MPI_Fint *ranks, MPI_Fint *newgroup, MPI_Fint *ierr)
{
    /* local variables */
    lam_group_t *c_group, *c_newgroup;

    /* make the fortran to c representation conversion */
    c_group = MPI_Group_f2c(*group);
    c_newgroup = MPI_Group_f2c(*newgroup);

    *ierr = MPI_Group_incl(c_group, *ranks, n, &c_newgroup);

    /* translate the results from c to fortran */
    *newgroup = c_newgroup->grp_f_to_c_index;
}
