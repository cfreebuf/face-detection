#ifndef _TENSORFLOW_MTCNN_H_
#define _TENSORFLOW_MTCNN_H_

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
// #include "tensorflow/c/c_api.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

#if 0
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"
#endif

struct FaceLandmark {
	float x[5];
	float y[5];
};

struct FaceBox {
	float x0;
	float y0;
	float x1;
	float y1;
	float score;
	float regress[4];
	float px0;
	float py0;
	float px1;
	float py1;
	FaceLandmark landmark;  
};

struct ScaleWindow {
	int h;
	int w;
	float scale;
};

class MtcnnTensorflow {
 public:
  int Init(const std::string& model_file);

  int PNet(cv::Mat& img, const ScaleWindow& win,
           std::vector<FaceBox>* output_boxes);
  int RNet(cv::Mat& img, std::vector<FaceBox>& pnet_boxes,
           std::vector<FaceBox>* output_boxes);
  int ONet(cv::Mat& img, std::vector<FaceBox>& rnet_boxes,
           std::vector<FaceBox>* output_boxes);

  void Detect(cv::Mat& src, std::vector<FaceBox>* face_boxs);

  void GenerateBoundingBox(const float* confidence_data, const float* reg_data,
                           float scale, float threshold, int h, int w,
                           bool transposed, std::vector<FaceBox>* output);

  void PyramidScales(int height, int width, int min_size, float factor,
                     std::vector<ScaleWindow>* ScaleWindows);

  void ProcessBoxes(std::vector<FaceBox>& input, int img_h, int img_w,
                    std::vector<FaceBox>* res);
  void Normalize(std::vector<FaceBox>& input, float threshold, int type,
                 std::vector<FaceBox>* output);
  void Regress(std::vector<FaceBox>* boxes);
  void Square(std::vector<FaceBox>* boxes);
  void Padding(int img_h, int img_w, std::vector<FaceBox>* res);

  void Patch(const cv::Mat& img, FaceBox& box, float* data_to, int height, int width);

 private:
  // TF_Session* session_;
  // TF_Graph* graph_;
  tensorflow::Session* session_;
};

#endif
