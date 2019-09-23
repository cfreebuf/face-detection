// File   common_gflags.cpp
// Author lidongming
// Date   2018-08-28 11:48:40
// Brief

#include "common/common_gflags.h"

// 0: detect loop 1: detect image 2: take photo
DEFINE_int32(detect_type, 0, "detect type");

// DEFINE_string(log_conf, "./conf/log.conf", "log conf");

DEFINE_string(facenet_server, "127.0.0.1:10101", "zmp server ip");
DEFINE_int32(facenet_client_timeout, 100, "timeout for facenet client");
DEFINE_string(face_infos_file, "./data/face_infos.json", "face infos");
DEFINE_double(min_face_dist, 1.1, "min face distance");

DEFINE_int32(camera_id, 0, "camera id");

DEFINE_string(mtcnn_model_file, "./models/mtcnn_frozen_model.pb", "mtcnn model");
DEFINE_string(facenet_model_file, "./models/facenet/embedding.pb", "facenet model");
DEFINE_string(test_images_path, "./data/test_images", "test images");

DEFINE_string(face_lmdb_path, "./data/face_lmdb", "lmdb file for faces");
DEFINE_int32(face_lmdb_size, 1024 * 1024 * 1024, "face lmdb file size");

DEFINE_string(face_index_file, "./data/face.index", "face index");
