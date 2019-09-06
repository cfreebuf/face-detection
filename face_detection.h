// File   face_detection.h
// Author lidongming1@360.cn
// Date   2019-08-22 16:35:13
// Brief

#ifndef _FACE_DETECTION_H_
#define _FACE_DETECTION_H_

#include <vector>
#include <memory>

#include "third_party/rapidjson/stringbuffer.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow_mtcnn.h"
#include "mtcnn.h"
#include "capture.h"
#include "util/util.h"
#include "util/base64.h"
#include "face_index.h"
#include "facenet_client.h"

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
  void Clean();

  void InitZmq();
  bool ConnectFaceNet();
  bool SendImage(const std::string& image_data);

  bool DetectLoop();
  bool DetectImage(const std::string& image_path);
  bool TakePhoto();

  int GetWarpAffineImage(cv::Mat& src,
                         const std::vector<cv::Point2f>& landmarks,
                         const cv::Size& dst_size, cv::Mat* dst);

  void WrapJson(const std::string& raw, std::string* data);

  void DrawRectAndLandmark(cv::Mat& frame, face_box& box);
  void DrawFaceInfo(cv::Mat& frame, FaceInfo& face_info, face_box& box, int i);

  cv::Mat NormalizedFaceRange(cv::Mat& frame, face_box& box);

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
	TF_Session* sess_;
	TF_Graph* graph_;
  FaceIndex face_index_;

  std::unique_ptr<FaceNetClient> facenet_client_;
};

#endif
