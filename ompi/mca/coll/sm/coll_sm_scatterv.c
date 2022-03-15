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

#include <string.h>

#include "coll_sm.h"
#include "ompi/communicator/communicator.h"
#include "ompi/constants.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/mca/coll/coll.h"
#include "opal/datatype/opal_convertor.h"
#include "opal/sys/atomic.h"

/*
 *	scatterv_intra
 *
 *	Function:	- scatterv operation
 *	Accepts:	- same arguments as MPI_Scatterv()
 *	Returns:	- MPI_SUCCESS or error code
 */
int mca_coll_sm_scatterv_intra(const void *send_buff, const int *send_counts,
                               const int *displs,
                               struct ompi_datatype_t *send_type,
                               void *recv_buff, int recv_count,
                               struct ompi_datatype_t *recv_type, int root,
                               struct ompi_communicator_t *comm,
                               mca_coll_base_module_t *module) {
  struct iovec iov;
  mca_coll_sm_module_t *sm_module = (mca_coll_sm_module_t *)module;
  mca_coll_sm_comm_t *data;
  int i;
  int ret;
  int comm_rank;
  int comm_size;
  unsigned int flag_num;
  unsigned int segment_num;
  unsigned int max_segment_num;
  mca_coll_sm_in_use_flag_t *flag;
  opal_convertor_t convertor;
  mca_coll_sm_data_index_t *index;

  char *send_buff_ptr_for_rank = NULL;
  size_t total_size = 0;
  size_t max_data = 0;
  size_t max_bytes = 0;
  size_t max_transfer_size = 0;

  /* Lazily enable the module the first time we invoke a collective
     on it */
  if (!sm_module->enabled) {
    if (OMPI_SUCCESS != (ret = ompi_coll_sm_lazy_enable(module, comm))) {
      return ret;
    }
  }
  data = sm_module->sm_comm_data;

  /* Setup some identities */
  comm_rank = ompi_comm_rank(comm);
  comm_size = ompi_comm_size(comm);

  iov.iov_len = mca_coll_sm_component.sm_fragment_size;

  size_t recv_type_size = recv_type->super.size;
  size_t recv_size = recv_count * recv_type_size;
  int fragment_set_size = mca_coll_sm_component.sm_fragment_size *
                          mca_coll_sm_component.sm_segs_per_inuse_flag;

  /* Correcting value for operations because count for each process
     may be different */
  int left_mcb_operation_count = 0;

  /*********************************************************************
   * Root
   *********************************************************************/

  if (root == comm_rank) {
    void *memory_map = NULL;
    opal_convertor_t *root_convertors_by_rank;
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

    /* Construct convertors. */
    for (int rank_iterator = 0; rank_iterator < comm_size; ++rank_iterator) {
      total_bytes_by_rank[rank_iterator] = 0;
      send_buff_ptr_for_rank =
          (char *)send_buff + send_type->super.size * displs[rank_iterator];

      if (comm_rank == rank_iterator) {
        /* Scatter for me. */
        memcpy(recv_buff, send_buff_ptr_for_rank, recv_size);
        total_bytes_by_rank[rank_iterator] =
            total_sizes_by_rank[rank_iterator] = recv_size;
        continue;
      }

      OBJ_CONSTRUCT(&root_convertors_by_rank[rank_iterator], opal_convertor_t);
      if (OMPI_SUCCESS !=
          (ret = opal_convertor_copy_and_prepare_for_send(
               ompi_mpi_local_convertor, &(send_type->super),
               send_counts[rank_iterator], send_buff_ptr_for_rank, 0,
               &root_convertors_by_rank[rank_iterator]))) {
        free(memory_map);
        return ret;
      }

      opal_convertor_get_packed_size(&root_convertors_by_rank[rank_iterator],
                                     &total_sizes_by_rank[rank_iterator]);
      total_size += total_sizes_by_rank[rank_iterator];

      size_t transfer_size = send_type->super.size * send_counts[rank_iterator];
      if (transfer_size > max_transfer_size) max_transfer_size = transfer_size;
    }

    left_mcb_operation_count = max_transfer_size / fragment_set_size +
                               (max_transfer_size % fragment_set_size ? 1 : 0);

    /* Scatter for others. */
    /* If we have data to process. Prevent zero-size. */
    if (max_bytes < total_size) {
      do {
        flag_num = (data->mcb_operation_count++ %
                    mca_coll_sm_component.sm_comm_num_in_use_flags);
        left_mcb_operation_count =
            left_mcb_operation_count == 0 ? 0 : left_mcb_operation_count - 1;

        /* Calculate process count for current mcb_operation_count.
           Not all process may need to transfer data and call FlAG_RELEASE.
           Always exclude root. */
        int processes_in_current_operation = comm_size;
        for (int target_rank = 0; target_rank < comm_size; ++target_rank) {
          if (total_bytes_by_rank[target_rank] ==
              total_sizes_by_rank[target_rank])
            --processes_in_current_operation;
        }

        FLAG_SETUP(flag_num, flag, data);
        FLAG_WAIT_FOR_IDLE(flag, scatterv_root_label);
        FLAG_RETAIN(flag, processes_in_current_operation,
                    data->mcb_operation_count - 1);

        /* Calculate start segment numbers range. */
        segment_num = flag_num * mca_coll_sm_component.sm_segs_per_inuse_flag;
        max_segment_num =
            (flag_num + 1) * mca_coll_sm_component.sm_segs_per_inuse_flag;
        do {
          /* Copy fragments into target_rank.s spaces. */
          for (int target_rank = 0; target_rank < comm_size; ++target_rank) {
            // If transmission already complete.
            if (total_bytes_by_rank[target_rank] ==
                total_sizes_by_rank[target_rank])
              continue;
            index = &(data->mcb_data_index[segment_num]);

            max_data = mca_coll_sm_component.sm_fragment_size;
            COPY_FRAGMENT_IN(root_convertors_by_rank[target_rank], index,
                             target_rank, iov, max_data);
            max_bytes += max_data;
            total_bytes_by_rank[target_rank] += max_data;

            /* Wait for write to absolutely complete */
            opal_atomic_wmb();

            /* Tell target_rank that this fragment is ready */
            NOTIFY_PROCESS_WITH_RANK(target_rank, index, max_data);
          }
          ++segment_num;
        } while (max_bytes < total_size && segment_num < max_segment_num);
      } while (max_bytes < total_size);
    }

    for (int r = 0; r < comm_size; ++r) {
      if (root != r) {
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
    if (OMPI_SUCCESS != (ret = opal_convertor_copy_and_prepare_for_recv(
                             ompi_mpi_local_convertor, &(recv_type->super),
                             recv_count, recv_buff, 0, &convertor))) {
      return ret;
    }
    opal_convertor_get_packed_size(&convertor, &total_size);
    max_transfer_size = recv_size;
    for (int rank_iterator = 0; rank_iterator < comm_size; ++rank_iterator) {
      if (root == rank_iterator) {
        continue;
      }
      size_t transfer_size = send_type->super.size * send_counts[rank_iterator];
      if (transfer_size > max_transfer_size) max_transfer_size = transfer_size;
    }
    left_mcb_operation_count = max_transfer_size / fragment_set_size +
                               (max_transfer_size % fragment_set_size ? 1 : 0);
    /* If we have data to process. Prevent zero-size. */
    if (max_bytes < total_size) {
      do {
        flag_num = (data->mcb_operation_count %
                    mca_coll_sm_component.sm_comm_num_in_use_flags);

        FLAG_SETUP(flag_num, flag, data);
        FLAG_WAIT_FOR_OP(flag, data->mcb_operation_count,
                         scatterv_nonroot_label1);

        ++data->mcb_operation_count;
        left_mcb_operation_count =
            left_mcb_operation_count == 0 ? 0 : left_mcb_operation_count - 1;

        segment_num = flag_num * mca_coll_sm_component.sm_segs_per_inuse_flag;
        max_segment_num =
            (flag_num + 1) * mca_coll_sm_component.sm_segs_per_inuse_flag;
        // if we already receive all data
        if (max_bytes < total_size) {
          do {
            index = &(data->mcb_data_index[segment_num]);

            /* Wait for my parent to tell me that the segment is ready */
            WAIT_FOR_NOTIFY(comm_rank, index, max_data,
                            scatterv_nonroot_label2);

            /* Copy to my output buffer */
            COPY_FRAGMENT_OUT(convertor, comm_rank, index, iov, max_data);

            max_bytes += max_data;
            ++segment_num;
          } while (max_bytes < total_size && segment_num < max_segment_num);
        }

        /* Wait for all copy-out writes to complete before I say
           I'm done with the segments */
        opal_atomic_wmb();

        /* We're finished with this set of segments */
        FLAG_RELEASE(flag);
      } while (max_bytes < total_size);
    }

    /* Kill the convertor */
    OBJ_DESTRUCT(&convertor);
  }

  /* Correct mcb_operation_count */

  data->mcb_operation_count += left_mcb_operation_count;

  /* All done */

  return OMPI_SUCCESS;
}
