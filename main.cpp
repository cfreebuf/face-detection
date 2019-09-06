// File   main.cpp
// Author lidongming1@360.cn
// Date   2019-08-22 16:17:17
// Brief

#include <thread>
#include <glog/logging.h>
#include "face_detection.h"
#include "facenet_adapter.h"
#include "common/common_gflags.h"

using namespace google;

// Reload gflags
std::unique_ptr<gflags::FlagSaver> current_flags(new gflags::FlagSaver());
void StartConfigReloadThread() {
    std::thread([]() {
        while (1) {
            current_flags.reset();
            current_flags.reset(new gflags::FlagSaver());
            gflags::ReparseCommandLineNonHelpFlags();
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }
    }).detach();
}

int main(int argc, char** argv) {
  // Init gflags
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::SetCommandLineOption("flagfile", "./conf/gflags.conf");
  StartConfigReloadThread();

  // Init glog
  google::InitGoogleLogging(argv[0]);
  FLAGS_log_dir = "./logs";
  // FLAGS_logtostderr = true;
  // FLAGS_alsologtostderr = true;
  FLAGS_log_prefix = true;
  FLAGS_logbufsecs = 0;

  LOG(INFO) << "Start face detection";

  google::FlushLogFiles(google::INFO);

  FaceDetection faced(FLAGS_detect_type);
  if (faced.Init() != 0) {
    LOG(FATAL) << "Failed to init face_detection";
    exit(-1);
  }

  faced.Start();

  return 0;
}
