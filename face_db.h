// File   face_db.h
// Author lidongming1@360.cn
// Date   2019-08-31 02:35:13
// Brief

#ifndef _FEATURE_DB_H_
#define _FEATURE_DB_H_

#include "util/rw_lock.h"
#include "lmdb_wrapper.h"

class FaceDB {
 public:
  static FaceDB& Instance() {
    static FaceDB instance;
    return instance;
  }

  FaceDB() { Init(); }
  ~FaceDB() {}

  int Init();

  bool Put(uint64_t face_id, const std::vector<double>& dim);
  bool Get(uint64_t face_id, std::vector<double>* dim);
  bool GetNext(lmdb::cursor& cursor, uint64_t* face_id, std::vector<double>* dim);
  void MakeValue(lmdb_wrapper::LMDBValue<double>& value, std::vector<double>* dim);

  lmdb::txn Txn() { return lmdb_env_->OpenReadTxn(); }

  lmdb::cursor Cursor(lmdb::txn& txn) { return lmdb_db_->cursor(txn); }

 private:
  std::unique_ptr<lmdb_wrapper::LMDBEnv> lmdb_env_;
  std::unique_ptr<lmdb_wrapper::LMDBDatabase<uint64_t, double>> lmdb_db_;

  SharedMutex mutex_;
};

#endif
