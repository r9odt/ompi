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
 *	allgather
 *
 *	Function:	- allgather
 *	Accepts:	- same as MPI_Allgather()
 *	Returns:	- MPI_SUCCESS or error code
 */
int mca_coll_sm_allgather_intra(const void *send_buff, int send_count,
                                struct ompi_datatype_t *send_type,
                                void *recv_buff, int recv_count,
                                struct ompi_datatype_t *recv_type,
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
  unsigned int out_segment_num;
  unsigned int in_segment_num;
  unsigned int max_segment_num;
  mca_coll_sm_in_use_flag_t *flag;
  opal_convertor_t convertor;
  mca_coll_sm_data_index_t *index;

  char *recv_buff_ptr_for_rank = NULL;
  size_t send_size = 0;
  size_t in_total_size = 0;
  size_t out_total_size = 0;
  size_t max_data = 0;
  size_t in_max_bytes = 0;
  size_t out_max_bytes = 0;

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

  void *memory_map = NULL;
  opal_convertor_t *out_convertors_by_rank = NULL;
  size_t *out_total_sizes_by_rank = NULL;
  size_t *out_total_bytes_by_rank = NULL;

  memory_map = (void *)malloc(comm_size *
                              (sizeof(opal_convertor_t) + 2 * sizeof(size_t)));
  if (!memory_map) {
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  out_total_sizes_by_rank = (size_t *)memory_map;
  out_total_bytes_by_rank =
      (size_t *)((size_t *)out_total_sizes_by_rank + comm_size);
  out_convertors_by_rank =
      (opal_convertor_t *)((size_t *)out_total_bytes_by_rank + comm_size);

  size_t recv_size = recv_type->super.size * recv_count;

  /* Construct convertors. */
  for (int rank_iterator = 0; rank_iterator < comm_size; ++rank_iterator) {
    out_total_bytes_by_rank[rank_iterator] = 0;
    recv_buff_ptr_for_rank = (char *)recv_buff + recv_size * rank_iterator;

    if (comm_rank == rank_iterator) {
      /* Gather for me. */
      if (send_buff == MPI_IN_PLACE) {
        out_total_bytes_by_rank[rank_iterator] =
            out_total_sizes_by_rank[rank_iterator] = 0;
        send_buff = recv_buff_ptr_for_rank;
        send_count = recv_count;
        send_type = recv_type;
      } else {
        send_size = send_count * send_type->super.size;
        out_total_bytes_by_rank[rank_iterator] =
            out_total_sizes_by_rank[rank_iterator] = send_size;
        memcpy(recv_buff_ptr_for_rank, send_buff, send_size);
      }
      continue;
    }

    OBJ_CONSTRUCT(&out_convertors_by_rank[rank_iterator], opal_convertor_t);
    if (OMPI_SUCCESS != (ret = opal_convertor_copy_and_prepare_for_recv(
                             ompi_mpi_local_convertor, &(recv_type->super),
                             recv_count, recv_buff_ptr_for_rank, 0,
                             &out_convertors_by_rank[rank_iterator]))) {
      free(memory_map);
      return ret;
    }

    opal_convertor_get_packed_size(&out_convertors_by_rank[rank_iterator],
                                   &out_total_sizes_by_rank[rank_iterator]);
    out_total_size += out_total_sizes_by_rank[rank_iterator];
  }

  OBJ_CONSTRUCT(&convertor, opal_convertor_t);
  if (OMPI_SUCCESS != (ret = opal_convertor_copy_and_prepare_for_send(
                           ompi_mpi_local_convertor, &(send_type->super),
                           send_count, send_buff, 0, &convertor))) {
    free(memory_map);
    return ret;
  }
  opal_convertor_get_packed_size(&convertor, &in_total_size);

  do {
    flag_num = (data->mcb_operation_count %
                mca_coll_sm_component.sm_comm_num_in_use_flags);

    FLAG_SETUP(flag_num, flag, data);

    /* Root always have id equal 0 */
    if (!comm_rank) {
      FLAG_WAIT_FOR_IDLE(flag, allgather_root_label1);
      FLAG_RETAIN(flag, comm_size, data->mcb_operation_count);
    } else {
      FLAG_WAIT_FOR_OP(flag, data->mcb_operation_count,
                       allgather_nonroot_label1);
    }
    ++data->mcb_operation_count;

    segment_num = flag_num * mca_coll_sm_component.sm_segs_per_inuse_flag;
    in_segment_num = segment_num;
    out_segment_num = segment_num;
    max_segment_num =
        (flag_num + 1) * mca_coll_sm_component.sm_segs_per_inuse_flag;

    while (in_max_bytes < in_total_size && in_segment_num < max_segment_num) {
      index = &(data->mcb_data_index[in_segment_num]);

      /* Copy to my shared segment. */
      max_data = mca_coll_sm_component.sm_fragment_size;
      COPY_FRAGMENT_IN(convertor, index, comm_rank, iov, max_data);
      in_max_bytes += max_data;

      /* Wait for write to absolutely complete */
      opal_atomic_wmb();

      NOTIFY_ALL_PROCESSES_WITHOUT_ME_EXTENDED(comm_rank, comm_size, index,
                                               max_data);
      ++in_segment_num;
    }

    while (out_max_bytes < out_total_size &&
           out_segment_num < max_segment_num) {
      index = &(data->mcb_data_index[out_segment_num]);

      /* Copy fragments from target_rank.s spaces. */
      for (int target_rank = 0; target_rank < comm_size; ++target_rank) {
        /* If transmission already complete. */
        if (out_total_bytes_by_rank[target_rank] ==
            out_total_sizes_by_rank[target_rank]) {
          continue;
        }
        WAIT_FOR_RANK_NOTIFY_EXTENDED(comm_rank, comm_size, target_rank, index,
                                      max_data, allgather_out);
        COPY_FRAGMENT_OUT(out_convertors_by_rank[target_rank], target_rank,
                          index, iov, max_data);
        out_total_bytes_by_rank[target_rank] += max_data;
        out_max_bytes += max_data;
      }
      ++out_segment_num;
    }
    /* Wait for all copy-out writes to complete */
    opal_atomic_wmb();
    FLAG_RELEASE(flag);
  } while ((out_max_bytes < out_total_size) && (in_max_bytes < in_total_size));

  for (int r = 0; r < comm_size; ++r) {
    if (comm_rank != r) {
      OBJ_DESTRUCT(&out_convertors_by_rank[r]);
    }
  }

  free(memory_map);

  /* Kill the convertor */
  OBJ_DESTRUCT(&convertor);

  /* All done */

  return OMPI_SUCCESS;
}
