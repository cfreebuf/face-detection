#include <gtest/gtest.h>
#include <iostream>
#include "face_db.h"

class FaceDBTest : public ::testing::Test {
 public:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(FaceDBTest, TestGetNext) {
  uint64_t face_id;
  std::vector<double> dims;

  FaceDB& face_db = FaceDB::Instance();

  lmdb::txn txn = face_db.Txn();
  lmdb::cursor cursor = face_db.Cursor(txn);

  while (face_db.GetNext(cursor, &face_id, &dims)) {
    std::cout << "face_id:" << face_id << std::endl;
    for (double v : dims) {
      std::cout << " " << v;
    }
    std::cout << std::endl;
  }
}
