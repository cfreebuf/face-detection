#ifndef _TENSORFLOW_MTCNN_H_
#define _TENSORFLOW_MTCNN_H_

#include <string>
#include <vector>
#include "tensorflow/c/c_api.h"
#include <opencv2/opencv.hpp>

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

class TFMtcnn {
 public:
  int Init(const std::string& model_file);

  void PNet(cv::Mat& img, const ScaleWindow& win,
            std::vector<FaceBox>* output_boxes);
  void RNet(cv::Mat& img, std::vector<FaceBox>& pnet_boxes,
            std::vector<FaceBox>* output_boxes);
  void ONet(cv::Mat& img, std::vector<FaceBox>& rnet_boxes,
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

  void CopyPatch(const cv::Mat& img, FaceBox& box, float* data_to,
                int height, int width);

 private:
  TF_Session* session_;
  TF_Graph* graph_;
};

#endif
