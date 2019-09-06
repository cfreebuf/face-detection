// File   common_gflags.h
// Author lidongming
// Date   2018-08-28 11:46:18
// Brief

#ifndef _COMMON_COMMON_GFLAGS_H_
#define _COMMON_COMMON_GFLAGS_H_

#include <gflags/gflags.h>

DECLARE_int32(detect_type);

// DECLARE_string(log_conf);

DECLARE_string(facenet_server);
DECLARE_int32(facenet_client_timeout);
DECLARE_string(face_infos_file);
DECLARE_double(min_face_dist);

DECLARE_string(zmq_facenet_server);
DECLARE_int32(zmq_recv_timeout);
DECLARE_int32(zmq_send_timeout);

DECLARE_int32(camera_id);

DECLARE_string(mtcnn_model_file);

DECLARE_string(test_image);

DECLARE_string(face_lmdb_path);
DECLARE_int32(face_lmdb_size);

DECLARE_string(face_index_file);

#endif  // _COMMON_COMMON_GFLAGS_H_
