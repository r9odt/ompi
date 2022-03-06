/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "coll_sm.h"
#include "ompi/communicator/communicator.h"
#include "ompi/constants.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi_config.h"

/*
 *      gather
 *
 *      Function:       - shared memory gather
 *      Accepts:        - same as MPI_Gather()
 *      Returns:        - MPI_SUCCESS or error code
 */
int mca_coll_sm_gather_intra(const void *send_buff, int send_count,
                             struct ompi_datatype_t *send_type, void *recv_buff,
                             int recv_count, struct ompi_datatype_t *recv_type,
                             int root, struct ompi_communicator_t *comm,
                             mca_coll_base_module_t *module) {
  struct iovec iov;
  mca_coll_sm_module_t *sm_module = (mca_coll_sm_module_t *)module;
  mca_coll_sm_comm_t *data;
  int ret;
  int comm_rank;
  int comm_size;
  unsigned int flag_num;
  unsigned int segment_num;
  unsigned int max_segment_num;
  mca_coll_sm_in_use_flag_t *flag;
  opal_convertor_t convertor;
  mca_coll_sm_data_index_t *index;

  char *recv_buff_ptr_for_rank = NULL;
  size_t send_type_size = 0;
  size_t send_size = 0;
  size_t total_size = 0;
  size_t max_data = 0;
  size_t max_bytes = 0;

  /* Lazily enable the module the first time we invoke a collective
     on it */
  if (!sm_module->enabled) {
    if (OMPI_SUCCESS != (ret = ompi_coll_sm_lazy_enable(module, comm))) {
      return ret;
    }
  }
  data = sm_module->sm_comm_data;

  // char sendtypename[MPI_MAX_OBJECT_NAME], recvtypename[MPI_MAX_OBJECT_NAME];
  // char commname[MPI_MAX_OBJECT_NAME];
  // int len;
  // int rank;

  // PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // PMPI_Type_get_name(send_type, sendtypename, &len);
  // PMPI_Type_get_name(recv_type, recvtypename, &len);
  // PMPI_Comm_get_name(comm, commname, &len);

  // fprintf(stderr,
  //         "@MPI_GATHER[%d]: root %d sendbuf %0" PRIxPTR
  //         " sendcount %d sendtype %s\n\trecvbuf %0" PRIxPTR
  //         " recvcount %d recvtype %s comm %s\n",
  //         rank, root, (uintptr_t)send_buff, send_count, sendtypename,
  //         (uintptr_t)recv_buff, recv_count, recvtypename, commname);
  // fflush(stderr);

  /* Setup some identities */
  comm_rank = ompi_comm_rank(comm);
  comm_size = ompi_comm_size(comm);
  // printf("Rank %d start gather\n", comm_rank);
  // fflush(stdout);
  iov.iov_len = mca_coll_sm_component.sm_fragment_size;

  /*********************************************************************
   * Root
   *********************************************************************/

  if (root == comm_rank) {
    void *memory_map = NULL;
    opal_convertor_t *root_convertors_by_rank = NULL;
    size_t *total_sizes_by_rank = NULL;
    size_t *total_bytes_by_rank = NULL;

    memory_map = (void *)malloc(
        comm_size * (sizeof(opal_convertor_t) + 2 * sizeof(size_t)));
    if (!memory_map) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }

    total_sizes_by_rank = (size_t *)memory_map;
    total_bytes_by_rank = (size_t *)((size_t *)total_sizes_by_rank + comm_size);
    root_convertors_by_rank =
        (opal_convertor_t *)((size_t *)total_bytes_by_rank + comm_size);

    size_t recv_size = recv_type->super.size * recv_count;

    /* Construct convertors. */
    for (int rank_iterator = 0; rank_iterator < comm_size; ++rank_iterator) {
      total_bytes_by_rank[rank_iterator] = 0;
      recv_buff_ptr_for_rank = (char *)recv_buff + recv_size * rank_iterator;

      if (comm_rank == rank_iterator) {
        /* Gather for me. */
        if (send_buff == MPI_IN_PLACE) {
          total_bytes_by_rank[rank_iterator] =
              total_sizes_by_rank[rank_iterator] = 0;
        } else {
          total_bytes_by_rank[rank_iterator] =
              total_sizes_by_rank[rank_iterator] = recv_size;
          memcpy(recv_buff_ptr_for_rank, send_buff, recv_size);
        }
        continue;
      }

      OBJ_CONSTRUCT(&root_convertors_by_rank[rank_iterator], opal_convertor_t);
      if (OMPI_SUCCESS != (ret = opal_convertor_copy_and_prepare_for_recv(
                               ompi_mpi_local_convertor, &(recv_type->super),
                               recv_count, recv_buff_ptr_for_rank, 0,
                               &root_convertors_by_rank[rank_iterator]))) {
        free(memory_map);
        return ret;
      }

      opal_convertor_get_packed_size(&root_convertors_by_rank[rank_iterator],
                                     &total_sizes_by_rank[rank_iterator]);
      total_size += total_sizes_by_rank[rank_iterator];
    }

    /* Gather for others. */
    do {
      flag_num = (data->mcb_operation_count++ %
                  mca_coll_sm_component.sm_comm_num_in_use_flags);

      // printf("Rank %d start gather FLAG_SETUP\n", comm_rank);
      FLAG_SETUP(flag_num, flag, data);
      // printf("Rank %d start gather FLAG_WAIT_FOR_IDLE %d\n",
      // flag->mcsiuf_num_procs_using);
      FLAG_WAIT_FOR_IDLE(flag, gather_root_label1);
      FLAG_RETAIN(flag, 1, data->mcb_operation_count - 1);
      // printf("Rank %d end gather FLAG_RETAIN\n", comm_rank);

      /* Calculate start segment numbers range. */
      segment_num = flag_num * mca_coll_sm_component.sm_segs_per_inuse_flag;
      max_segment_num =
          (flag_num + 1) * mca_coll_sm_component.sm_segs_per_inuse_flag;
      do {
        index = &(data->mcb_data_index[segment_num]);
        // TODO: Implement order as-is by ready non-roots. Notify count may be
        // different.
        /* Copy fragments into target_rank.s spaces. */
        for (int target_rank = 0; target_rank < comm_size; ++target_rank) {
          /* If transmission already complete. */
          if (total_bytes_by_rank[target_rank] ==
              total_sizes_by_rank[target_rank]) {
            continue;
          }

          /* Wait for notify from non-roots. */
          // printf("Rank %d start gather WAIT_FOR_NOTIFY\n", comm_rank);
          WAIT_FOR_NOTIFY(target_rank, index, max_data, gather_root_label2);
          // printf("Rank %d end gather WAIT_FOR_NOTIFY\n", comm_rank);

          /* Copy to my output buffer */
          COPY_FRAGMENT_OUT(root_convertors_by_rank[target_rank], target_rank,
                            index, iov, max_data);
          max_bytes += max_data;
          total_bytes_by_rank[target_rank] += max_data;
        }
        ++segment_num;
      } while (max_bytes < total_size && segment_num < max_segment_num);

      /* Wait for all copy-out writes to complete */
      opal_atomic_wmb();

      /* We're finished with this set of segments */
      FLAG_RELEASE(flag);
    } while (max_bytes < total_size);

    for (int r = 0; r < comm_size; ++r) {
      if (comm_rank != r) {
        OBJ_DESTRUCT(&root_convertors_by_rank[r]);
      }
    }

    free(memory_map);
  }

  /*********************************************************************
   * Non-root
   *********************************************************************/

  else {
    OBJ_CONSTRUCT(&convertor, opal_convertor_t);
    if (OMPI_SUCCESS != (ret = opal_convertor_copy_and_prepare_for_send(
                             ompi_mpi_local_convertor, &(send_type->super),
                             send_count, send_buff, 0, &convertor))) {
      return ret;
    }
    opal_convertor_get_packed_size(&convertor, &total_size);

    do {
      flag_num = (data->mcb_operation_count %
                  mca_coll_sm_component.sm_comm_num_in_use_flags);

      // printf("Rank %d start gather FLAG_SETUP\n", comm_rank);
      FLAG_SETUP(flag_num, flag, data);
      FLAG_WAIT_FOR_OP(flag, data->mcb_operation_count, gather_nonroot_label1);
      // printf("Rank %d start gather FLAG_WAIT_FOR_OP\n", comm_rank);
      ++data->mcb_operation_count;

      segment_num = flag_num * mca_coll_sm_component.sm_segs_per_inuse_flag;
      max_segment_num =
          (flag_num + 1) * mca_coll_sm_component.sm_segs_per_inuse_flag;
      do {
        index = &(data->mcb_data_index[segment_num]);

        /* Copy to my shared segment. */
        max_data = mca_coll_sm_component.sm_fragment_size;
        COPY_FRAGMENT_IN(convertor, index, comm_rank, iov, max_data);
        max_bytes += max_data;

        /* Wait for write to absolutely complete */
        opal_atomic_wmb();

        /* Notify root that fragment is ready.
           Root get notification from my segment*/
        // printf("Rank %d start gather NOTIFY_PROCESS_WITH_RANK %d %p %d
        // %d/%d\n", comm_rank,send_count, send_buff, segment_num, max_bytes,
        // total_size);
        NOTIFY_PROCESS_WITH_RANK(comm_rank, index, max_data);
        // printf("Rank %d end gather NOTIFY_PROCESS_WITH_RANK %d %d/%d\n",
        // comm_rank, segment_num, max_bytes, total_size);
        ++segment_num;
      } while (max_bytes < total_size && segment_num < max_segment_num);
    } while (max_bytes < total_size);
    /* Kill the convertor */
    OBJ_DESTRUCT(&convertor);
  }

  /* All done */

  // printf("Rank %d end gather\n", comm_rank);
  // fflush(stdout);
  return OMPI_SUCCESS;
}
