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
 *	gatherv_intra
 *
 *	Function:	- basic gatherv operation
 *	Accepts:	- same arguments as MPI_Gatherv()
 *	Returns:	- MPI_SUCCESS or error code
 */
int mca_coll_sm_gatherv_intra(const void *send_buff, int send_count,
                              struct ompi_datatype_t *send_type,
                              void *recv_buff, const int *recv_counts,
                              const int *displs,
                              struct ompi_datatype_t *recv_type, int root,
                              struct ompi_communicator_t *comm,
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

  int counts_sended = 0;
  char *recv_buff_ptr_for_rank = NULL;
  size_t send_type_size = 0;
  size_t send_size = 0;
  size_t total_size = 0;
  size_t max_data = 0;
  size_t max_bytes = 0;
  size_t max_transfer_size = 0;

  /*
   * Lazily enable the module the first time we invoke a collective on it.
   */
  if (!sm_module->enabled) {
    if (OMPI_SUCCESS != (ret = ompi_coll_sm_lazy_enable(module, comm))) {
      return ret;
    }
  }
  data = sm_module->sm_comm_data;

  /*
   * Setup some identities.
   */
  comm_rank = ompi_comm_rank(comm);
  comm_size = ompi_comm_size(comm);

  iov.iov_len = mca_coll_sm_component.sm_fragment_size;

  int fragment_set_size = mca_coll_sm_component.sm_fragment_size *
                          mca_coll_sm_component.sm_segs_per_inuse_flag;

  /*
   * Correcting value for operations because count for each process
   * may be different.
   */
  int left_mcb_operation_count = 0;

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

    /*
     * Construct convertors.
     */
    for (int rank_iterator = 0; rank_iterator < comm_size; ++rank_iterator) {
      total_bytes_by_rank[rank_iterator] = 0;
      recv_buff_ptr_for_rank =
          (char *)recv_buff + recv_type->super.size * displs[rank_iterator];

      if (comm_rank == rank_iterator) {
        /*
         * Gather for me.
         */
        if (send_buff == MPI_IN_PLACE) {
          total_bytes_by_rank[rank_iterator] =
              total_sizes_by_rank[rank_iterator] = 0;
        } else {
          size_t recv_size = recv_counts[rank_iterator] * recv_type->super.size;
          total_bytes_by_rank[rank_iterator] =
              total_sizes_by_rank[rank_iterator] = recv_size;
          // TODO: user-defined datatypes not compatible with base memcpy
          memcpy(recv_buff_ptr_for_rank, send_buff, recv_size);
        }
        continue;
      }

      OBJ_CONSTRUCT(&root_convertors_by_rank[rank_iterator], opal_convertor_t);
      if (OMPI_SUCCESS !=
          (ret = opal_convertor_copy_and_prepare_for_recv(
               ompi_mpi_local_convertor, &(recv_type->super),
               recv_counts[rank_iterator], recv_buff_ptr_for_rank, 0,
               &root_convertors_by_rank[rank_iterator]))) {
        free(memory_map);
        return ret;
      }

      opal_convertor_get_packed_size(&root_convertors_by_rank[rank_iterator],
                                     &total_sizes_by_rank[rank_iterator]);
      total_size += total_sizes_by_rank[rank_iterator];

      size_t transfer_size = recv_type->super.size * recv_counts[rank_iterator];
      if (transfer_size > max_transfer_size) max_transfer_size = transfer_size;
    }
    left_mcb_operation_count = max_transfer_size / fragment_set_size +
                               (max_transfer_size % fragment_set_size ? 1 : 0);

    /*
     * Gather.
     */
    do {
      flag_num = (data->mcb_operation_count++ %
                  mca_coll_sm_component.sm_comm_num_in_use_flags);
      left_mcb_operation_count =
          left_mcb_operation_count == 0 ? 0 : left_mcb_operation_count - 1;

      FLAG_SETUP(flag_num, flag, data);
      FLAG_WAIT_FOR_IDLE(flag, gatherv_root_label1);
      FLAG_RETAIN(flag, comm_size, data->mcb_operation_count - 1);

      /*
       * Calculate start segment numbers range.
       */
      segment_num = flag_num * mca_coll_sm_component.sm_segs_per_inuse_flag;
      max_segment_num =
          (flag_num + 1) * mca_coll_sm_component.sm_segs_per_inuse_flag;

      if (!counts_sended) {
        flag->stype_size = recv_type->super.size;
        memcpy((char *)flag + flag->ssizes_shift, recv_counts,
               sizeof(int) * comm_size);
        int *ssize_notify = (int *)((char *)flag + flag->ssizes_shift +
                                    sizeof(int) * comm_size);
        for (int target_rank = 0; target_rank < comm_size; ++target_rank) {
          if (root == target_rank) {
            continue;
          }
          ssize_notify[target_rank] = 1;
        }
        opal_atomic_wmb();

        counts_sended = 1;
      }

      while (max_bytes < total_size && segment_num < max_segment_num) {
        // TODO: Implement order as-is by ready non-roots. Notify count may be
        // different.
        /*
         * Copy fragments into target_rank.s spaces.
         */
        for (int target_rank = 0; target_rank < comm_size; ++target_rank) {
          /*
           * If transmission already complete.
           */
          if (total_bytes_by_rank[target_rank] ==
              total_sizes_by_rank[target_rank]) {
            continue;
          }

          index = &(data->mcb_data_index[segment_num]);

          /*
           * Wait for notify from non-roots.
           */
          WAIT_FOR_NOTIFY(target_rank, index, max_data, gatherv_root_label2);

          /*
           * Copy to my output buffer.
           */
          COPY_FRAGMENT_OUT(root_convertors_by_rank[target_rank], target_rank,
                            index, iov, max_data);

          opal_atomic_wmb();

          max_bytes += max_data;
          total_bytes_by_rank[target_rank] += max_data;
        }
        ++segment_num;
      }

      /*
       * Wait for all copy-out writes to complete.
       */
      opal_atomic_wmb();

      /*
       * We're finished with this set of segments.
       */
      FLAG_RELEASE(flag);
    } while (max_bytes < total_size && left_mcb_operation_count > 0);

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
    int *_recv_counts = NULL;
    _recv_counts = (int *)malloc(comm_size * sizeof(int));
    size_t _recv_type_size = 0;
    if (!_recv_counts) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }
    OBJ_CONSTRUCT(&convertor, opal_convertor_t);
    if (OMPI_SUCCESS != (ret = opal_convertor_copy_and_prepare_for_send(
                             ompi_mpi_local_convertor, &(send_type->super),
                             send_count, send_buff, 0, &convertor))) {
      return ret;
    }
    opal_convertor_get_packed_size(&convertor, &total_size);

    send_type_size = send_type->super.size;
    send_size = send_count * send_type_size;

    max_transfer_size = send_size;
    _recv_counts[root] = 0;

    do {
      flag_num = (data->mcb_operation_count %
                  mca_coll_sm_component.sm_comm_num_in_use_flags);

      FLAG_SETUP(flag_num, flag, data);
      FLAG_WAIT_FOR_OP(flag, data->mcb_operation_count, gatherv_nonroot_label1);

      ++data->mcb_operation_count;

      segment_num = flag_num * mca_coll_sm_component.sm_segs_per_inuse_flag;
      max_segment_num =
          (flag_num + 1) * mca_coll_sm_component.sm_segs_per_inuse_flag;

      if (!counts_sended) {
        int *ssize_notify = (int *)((char *)flag + flag->ssizes_shift +
                                    sizeof(int) * comm_size);
        SPIN_CONDITION(ssize_notify[comm_rank] != 0,
                       gatherv_nonroot_counts_sync);

        ssize_notify[comm_rank] = 0;
        _recv_type_size = flag->stype_size;
        memcpy(_recv_counts, (char *)flag + flag->ssizes_shift,
               sizeof(int) * comm_size);
        opal_atomic_wmb();

        left_mcb_operation_count =
            max_transfer_size / fragment_set_size +
            (max_transfer_size % fragment_set_size ? 1 : 0);

        counts_sended = 1;
      }
      left_mcb_operation_count =
          left_mcb_operation_count == 0 ? 0 : left_mcb_operation_count - 1;

      while (max_bytes < total_size && segment_num < max_segment_num) {
        index = &(data->mcb_data_index[segment_num]);

        /*
         * Copy to my shared segment.
         */
        max_data = mca_coll_sm_component.sm_fragment_size;
        COPY_FRAGMENT_IN(convertor, index, comm_rank, iov, max_data);
        max_bytes += max_data;

        /*
         * Wait for write to absolutely complete.
         */
        opal_atomic_wmb();

        /*
         * Notify root that fragment is ready.
         * Root get notification from my segment.
         */
        NOTIFY_PROCESS_WITH_RANK(comm_rank, index, max_data);

        opal_atomic_wmb();

        ++segment_num;
      }

      /*
       * We're finished with this set of segments.
       */
      FLAG_RELEASE(flag);
    } while (max_bytes < total_size && left_mcb_operation_count > 0);

    /*
     * Kill the convertor.
     */
    OBJ_DESTRUCT(&convertor);
  }

  /*
   * All done.
   */

  return OMPI_SUCCESS;
}
