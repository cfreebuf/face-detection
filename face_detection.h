// File   face_detection.h
// Author lidongming1@360.cn
// Date   2019-08-22 16:35:13
// Brief

#ifndef _FACE_DETECTION_H_
#define _FACE_DETECTION_H_

#include <vector>
#include <memory>

#include "third_party/rapidjson/stringbuffer.h"
#include "mtcnn_tensorflow.h"
#include "facenet_tensorflow.h"
#include "capture.h"
#include "util/util.h"
#include "util/base64.h"
#include "face_index.h"

enum DetectType {
  DETECT_LOOP = 0,
  DETECT_BY_IMAGE,
  DETECT_TAKE_PHOTO
};

class FaceDetection {
 public:
  FaceDetection(int detect_type);
  ~FaceDetection();

  int Init();
  int InitCapture();

  bool Start();
  void Stop();

  void InitZmq();
  bool ConnectFaceNet();
  bool SendImage(const std::string& image_data);

  bool DetectLoop();
  bool DetectImages();
  bool TakePhoto();
  bool DetectImage(const std::string& image_path);

  int GetWarpAffineImage(cv::Mat& src,
                         const std::vector<cv::Point2f>& landmarks,
                         const cv::Size& dst_size, cv::Mat* dst);

  void WrapJson(const std::string& raw, std::string* data);

  void DrawRectAndLandmark(cv::Mat& frame, FaceBox& box);
  void DrawFaceInfo(cv::Mat& frame, FaceInfo& face_info, FaceBox& box, int i);

  cv::Mat NormalizedFaceRange(cv::Mat& frame, FaceBox& box);

  bool GetFaceDims(cv::Mat& face_img, std::vector<double>* dim);

  static int Mat2Base64(const cv::Mat& img, std::string img_type, std::string* data) {
  	std::string img_data;
  	std::vector<uchar> img_array;
  	cv::imencode(img_type, img, img_array);
  	util::Base64Encode(img_array.data(), img_array.size(), data);
  	return 0;
  }

 private:
  DetectType type_;
  std::shared_ptr<Capture> capture_;
  MtcnnTensorflow mtcnn_tensorflow_;
  FaceNetTensorflow facenet_tensorflow_;
  FaceIndex face_index_;
};

#endif
