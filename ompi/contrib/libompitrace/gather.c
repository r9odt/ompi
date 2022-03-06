/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2008 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2009      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2013      Los Alamos National Security, LLC.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include <stdio.h>

#include "ompi/mpi/c/bindings.h"
#include "ompi_config.h"
#include "opal_stdint.h"

int MPI_Gather(const void *send_buff, int send_count, MPI_Datatype send_type,
               void *recv_buff, int recv_count, MPI_Datatype recv_type,
               int root, MPI_Comm comm) {
  char sendtypename[MPI_MAX_OBJECT_NAME], recvtypename[MPI_MAX_OBJECT_NAME];
  char commname[MPI_MAX_OBJECT_NAME];
  int len;
  int rank;

  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Type_get_name(send_type, sendtypename, &len);
  PMPI_Type_get_name(recv_type, recvtypename, &len);
  PMPI_Comm_get_name(comm, commname, &len);

  fprintf(stderr,
          "MPI_GATHER[%d]: root %d sendbuf %0" PRIxPTR
          " sendcount %d sendtype %s\n\trecvbuf %0" PRIxPTR
          " recvcount %d recvtype %s comm %s\n",
          rank, root, (uintptr_t)send_buff, send_count, sendtypename,
          (uintptr_t)recv_buff, recv_count, recvtypename, commname);
  fflush(stderr);

  return PMPI_Gather(send_buff, send_count, send_type, recv_buff, recv_count,
                     recv_type, root, comm);
}
