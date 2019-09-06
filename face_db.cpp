// File   face_db.cpp
// Author lidongming1@360.cn
// Date   2019-08-30 19:29:00
// Brief

#include <glog/logging.h>
#include <iostream>
#include "face_db.h"
#include "common/common_gflags.h"

int FaceDB::Init() {
  LOG(INFO) << "Start init face db size:" << FLAGS_face_lmdb_size;

  lmdb_env_ = std::make_unique<lmdb_wrapper::LMDBEnv>(FLAGS_face_lmdb_path.c_str());
  lmdb_db_ = std::make_unique<lmdb_wrapper::LMDBDatabase<uint64_t, double>>("faces");
  LOG(INFO) << "Finish init face db";
  return 0;
}

bool FaceDB::Put(uint64_t face_id, const std::vector<double>& dim) {
  WriteLock lock(mutex_);
  auto txn = lmdb_env_->OpenWriteTxn();
  bool ret = lmdb_db_->put(txn, face_id, &dim[0], dim.size());
  if (ret) {
    txn.commit();
  }
  return ret;
}

void FaceDB::MakeValue(lmdb_wrapper::LMDBValue<double>& value,
                       std::vector<double>* dim) {
    for (int i = 0; i < value.size(); i++) {
      dim->push_back(value[i]);
    }
}

bool FaceDB::Get(uint64_t face_id, std::vector<double>* dim) {
  ReadLock lock(mutex_);
  lmdb_wrapper::LMDBValue<double> value;
  auto txn = lmdb_env_->OpenReadTxn();
  bool ret = lmdb_db_->get(txn, face_id, &value);
  if (ret) {
    MakeValue(value, dim);
  }
  return ret;
}

bool FaceDB::GetNext(lmdb::cursor& cursor, uint64_t* face_id,
                     std::vector<double>* dim) {
  ReadLock lock(mutex_);

  lmdb_wrapper::LMDBValue<uint64_t> key;
  lmdb_wrapper::LMDBValue<double> value;

  bool ret = lmdb_db_->get_next(cursor, &key, &value);
  if (ret) {
    *face_id = (uint64_t)(*(uint64_t*)&key[0]);
    MakeValue(value, dim);
  }
  return ret;
}
