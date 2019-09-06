// File   main.cpp
// Author lidongming1@360.cn
// Date   2019-08-31 11:55:18
// Brief

#include <gtest/gtest.h>
#include "common/common_gflags.h"
#include <glog/logging.h>

using namespace google;

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_log_dir = "./logs";
  google::FlushLogFiles(google::INFO);
  return RUN_ALL_TESTS();
}
