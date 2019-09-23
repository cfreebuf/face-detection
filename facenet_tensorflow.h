// File   facenet_tensorflow.h
// Author lidongming1@360.cn
// Date   2019-09-20 17:45:05
// Brief

#ifndef _TENSORFLOW_FACENET_H_
#define _TENSORFLOW_FACENET_H_

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

class FaceNetTensorflow {
 public:
  int Init(const std::string& model_file);

  int GenerateEmbedding(std::vector<cv::Mat>& faces,
                        std::vector<std::vector<double>>* dims);

 private:
  tensorflow::Session* session_;
};

#endif
