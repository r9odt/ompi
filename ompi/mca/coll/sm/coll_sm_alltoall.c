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
 *	alltoall_intra
 *
 *	Function:	- MPI_Alltoall
 *	Accepts:	- same as MPI_Alltoall()
 *	Returns:	- MPI_SUCCESS or an MPI error code
 */

/*
sbuf          rbuf
a0b0c0d0      a0a1a2a3
a1b1c1d1  --\ b0b1b2b3
a2b2c2d2  --/ c0c1c2c3
a3b3c3d3      d0d1d2d3

*/

int mca_coll_sm_alltoall_intra(const void *send_buff, int send_count,
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
  mca_coll_sm_data_index_t *index;

  char *recv_buff_ptr_for_rank = NULL;
  char *send_buff_ptr_for_rank = NULL;
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
  opal_convertor_t *in_convertors_by_rank = NULL;
  size_t *in_total_sizes_by_rank = NULL;
  size_t *in_total_bytes_by_rank = NULL;

  memory_map = (void *)malloc(comm_size * 2 *
                              (sizeof(opal_convertor_t) + 2 * sizeof(size_t)));
  if (!memory_map) {
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  out_total_sizes_by_rank = (size_t *)memory_map;
  out_total_bytes_by_rank =
      (size_t *)((size_t *)out_total_sizes_by_rank + comm_size);
  in_total_sizes_by_rank =
      (size_t *)((size_t *)out_total_bytes_by_rank + comm_size);
  in_total_bytes_by_rank =
      (size_t *)((size_t *)in_total_sizes_by_rank + comm_size);
  out_convertors_by_rank =
      (opal_convertor_t *)((size_t *)in_total_bytes_by_rank + comm_size);
  in_convertors_by_rank =
      (opal_convertor_t *)((opal_convertor_t *)out_convertors_by_rank +
                           comm_size);

  if (send_buff == MPI_IN_PLACE) {
    send_buff = recv_buff;
    send_count = recv_count;
    send_type = recv_type;
  }
  size_t recv_size = recv_type->super.size * recv_count;
  size_t send_size = send_type->super.size * send_count;

  printf("Rank %d init\n", comm_rank);
  /* Construct convertors. */
  for (int rank_iterator = 0; rank_iterator < comm_size; ++rank_iterator) {
    out_total_bytes_by_rank[rank_iterator] = 0;
    in_total_bytes_by_rank[rank_iterator] = 0;
    recv_buff_ptr_for_rank = (char *)recv_buff + recv_size * rank_iterator;
    send_buff_ptr_for_rank = (char *)send_buff + send_size * rank_iterator;

    if (comm_rank == rank_iterator) {
      /* Copy for me. */
      out_total_bytes_by_rank[rank_iterator] =
          out_total_sizes_by_rank[rank_iterator] = recv_size;
      in_total_bytes_by_rank[rank_iterator] =
          in_total_sizes_by_rank[rank_iterator] = send_size;
      if (send_buff == MPI_IN_PLACE) {
      } else {
        memcpy(recv_buff_ptr_for_rank, send_buff_ptr_for_rank, send_size);
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

    OBJ_CONSTRUCT(&in_convertors_by_rank[rank_iterator], opal_convertor_t);
    if (OMPI_SUCCESS != (ret = opal_convertor_copy_and_prepare_for_recv(
                             ompi_mpi_local_convertor, &(send_type->super),
                             send_count, send_buff_ptr_for_rank, 0,
                             &in_convertors_by_rank[rank_iterator]))) {
      free(memory_map);
      return ret;
    }

    opal_convertor_get_packed_size(&in_convertors_by_rank[rank_iterator],
                                   &in_total_sizes_by_rank[rank_iterator]);
    in_total_size += in_total_sizes_by_rank[rank_iterator];
  }

  int in_current_processing_rank = 0;
  int out_current_processing_rank = 0;
  printf("Rank %d start\n", comm_rank);
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
    if (in_max_bytes < in_total_size) {
      in_current_processing_rank = 0;
      do {
        index = &(data->mcb_data_index[in_segment_num]);
        printf("Rank %d check in_current_processing_rank %d %d/%d\n", comm_rank,
               in_current_processing_rank,
               in_total_bytes_by_rank[in_current_processing_rank],
               in_total_sizes_by_rank[in_current_processing_rank]);
        /* If transmission for this rank already complete */
        if (in_total_bytes_by_rank[in_current_processing_rank] ==
            in_total_sizes_by_rank[in_current_processing_rank]) {
          printf("Rank %d ++in_current_processing_rank\n", comm_rank);
          ++in_current_processing_rank;
          continue;
        }
        printf("Rank %d end check in_current_processing_rank %d\n", comm_rank,
               in_current_processing_rank);

        /* Copy to my shared segment. */
        max_data = mca_coll_sm_component.sm_fragment_size;
        COPY_FRAGMENT_IN(in_convertors_by_rank[in_current_processing_rank],
                         index, comm_rank, iov, max_data);
        in_total_bytes_by_rank[in_current_processing_rank] += max_data;
        in_max_bytes += max_data;

        /* Wait for write to absolutely complete */
        opal_atomic_wmb();

        printf("Rank %d NOTIFY_PROCESS_WITH_RANK_EXTENDED %d seg %d\n",
               comm_rank, in_current_processing_rank, in_segment_num);
        NOTIFY_PROCESS_WITH_RANK_EXTENDED(
            comm_rank, comm_size, in_current_processing_rank, index, max_data);
        ++in_segment_num;
      } while (in_max_bytes < in_total_size &&
               in_segment_num < max_segment_num);
    }

    if (out_max_bytes < out_total_size) {
      do {
        index = &(data->mcb_data_index[out_segment_num]);
        printf("Rank %d check out_current_processing_rank %d\n", comm_rank,
               out_current_processing_rank);
        /* If transmission for this rank already complete */
        if (out_total_bytes_by_rank[out_current_processing_rank] ==
            out_total_sizes_by_rank[out_current_processing_rank]) {
          printf("Rank %d ++out_current_processing_rank\n", comm_rank);
          ++out_current_processing_rank;
          continue;
        }
        printf("Rank %d end check out_current_processing_rank %d\n", comm_rank,
               out_current_processing_rank);

        /* Copy fragments from target_rank.s spaces. */
        printf("Rank %d WAIT_FOR_RANK_NOTIFY_EXTENDED %d seg %d\n", comm_rank,
               out_current_processing_rank, out_segment_num);
        WAIT_FOR_RANK_NOTIFY_EXTENDED(comm_rank, comm_size,
                                      out_current_processing_rank, index,
                                      max_data, alltoall_out);
        printf("Rank %d COPY_FRAGMENT_OUT\n", comm_rank);
        COPY_FRAGMENT_OUT(out_convertors_by_rank[out_current_processing_rank],
                          out_current_processing_rank, index, iov, max_data);
        printf("Rank %d out_total_bytes_by_rank +=\n", comm_rank);
        out_total_bytes_by_rank[out_current_processing_rank] += max_data;
        printf("Rank %d out_total_bytes_by_rank += end\n", comm_rank);
        out_max_bytes += max_data;
        ++out_segment_num;
      } while (out_max_bytes < out_total_size &&
               out_segment_num < max_segment_num);
    }
    /* Wait for all copy-out writes to complete */
    opal_atomic_wmb();
    FLAG_RELEASE(flag);
  } while ((out_max_bytes < out_total_size) && (in_max_bytes < in_total_size));

  printf("Rank %d Release convertors_by_rank\n", comm_rank);
  for (int r = 0; r < comm_size; ++r) {
    if (comm_rank != r) {
      OBJ_DESTRUCT(&out_convertors_by_rank[r]);
      OBJ_DESTRUCT(&in_convertors_by_rank[r]);
    }
  }

  free(memory_map);

  /* All done */

  return OMPI_SUCCESS;
}
