// File   camera.cpp
// Author lidongming1@360.cn
// Date   2019-08-22 15:15:45
// Brief

#ifndef _CAPTURE_H_
#define _CAPTURE_H_

#include <iostream>
#include <opencv2/opencv.hpp>

class Capture {
 public:
  explicit Capture(int dev) : init_(false) {
    capture_.open(dev);
    if (!capture_.isOpened()) {
      std::cerr << "failed to open camera " << dev << std::endl;
    } else {
      init_ = true;
    }
  }

  cv::VideoCapture* capture() {
    if (init_) {
      return &capture_;
    } else {
      return NULL;
    }
  }

  bool is_inited() {
    return init_;
  }

  void GetFrame(cv::Mat* frame) {
    capture_.read(*frame);
  }

 private:
  cv::VideoCapture capture_;
  bool init_;
};

#endif
