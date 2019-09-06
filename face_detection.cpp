// File   face_detection.cpp
// Author lidongming1@360.cn
// Date   2019-08-22 16:08:41
// Brief

#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <glog/logging.h>

#include "third_party/rapidjson/rapidjson.h"
#include "third_party/rapidjson/document.h"
#include "third_party/rapidjson/writer.h"

#include "face_detection.h"
#include "face_db.h"
#include "common/common_gflags.h"
#include "util/util.h"
#include "util/base64.h"
#include "util/opencv_util.h"
#include "util/string_utils.h"

FaceDetection::FaceDetection(int type) : type_(DetectType(type)) {
  LOG(INFO) << "Start init mtcnn";
  sess_ = load_graph(FLAGS_mtcnn_model_file.c_str(), &graph_);
  if (sess_ == NULL) {
    LOG(WARNING) << "Failed to load graph for mtcnn, model:"
                 << FLAGS_mtcnn_model_file;
    exit(-1);
  }
  LOG(INFO) << "Finish init mtcnn";

  facenet_client_ = std::make_unique<FaceNetClient>(grpc::CreateChannel(
      FLAGS_facenet_server, grpc::InsecureChannelCredentials()));
}

FaceDetection::~FaceDetection() {
  if (sess_ != NULL || graph_ != NULL) {
    Clean();
  }
}

int FaceDetection::Init() {
  LOG(INFO) << "Start init face_detection";
  if (type_ == DETECT_LOOP || type_ == DETECT_TAKE_PHOTO) {
    LOG(INFO) << "Start init camera";
    if (InitCapture() != 0) {
      return -1;
    }
    LOG(INFO) << "Finish init camera";
  }
  LOG(INFO) << "Finish init face_detection";
  return 0;
}

int FaceDetection::InitCapture() {
  capture_ = std::make_shared<Capture>(FLAGS_camera_id);
  if (capture_ == nullptr) {
    LOG(WARNING) << "Failed to init camera camera_id:" << FLAGS_camera_id;
    return -1;
  }
  return 0;
}

void FaceDetection::DrawRectAndLandmark(cv::Mat& frame, face_box& box) {
  cv::rectangle(frame, cv::Point(box.x0, box.y0), cv::Point(box.x1, box.y1),
                cv::Scalar(0, 255, 0), 1);

  for (int l = 0; l < 5; l++) {
		cv::circle(frame, cv::Point(box.landmark.x[l], box.landmark.y[l]), 1,
               cv::Scalar(0, 0, 255), 2);
	}
}

void FaceDetection::DrawFaceInfo(cv::Mat& frame, FaceInfo& face_info,
                                 face_box& box, int i) {
  const int font_height = 30;
  const int box_width = 150;
  const int box_height = 4 * font_height + 10;
  int sx = 0;
  int sy = 35 + i * box_height;
  int ex = box_width;
  int ey = 35 + (i + 1) * box_height;

  // Draw info box
  cv::rectangle(frame, cv::Point(sx, sy), cv::Point(ex, ey),
                cv::Scalar(10, 10, 10), 1);

  // Draw face info
  cv::putText(frame, "Name:" + face_info.name,
              cv::Point(0, sy + 1 * font_height), cv::FONT_HERSHEY_COMPLEX, 0.5,
              cv::Scalar(10, 10, 110));
  cv::putText(frame, "Age:" + std::to_string(face_info.age),
              cv::Point(0, sy + 2 * font_height), cv::FONT_HERSHEY_COMPLEX, 0.5,
              cv::Scalar(10, 10, 110));
  cv::putText(frame, "FaceID:" + std::to_string(face_info.face_id),
              cv::Point(0, sy + 3 * font_height), cv::FONT_HERSHEY_COMPLEX, 0.5,
              cv::Scalar(10, 10, 110));
  cv::putText(frame, "Gender:" + std::to_string(face_info.gender),
              cv::Point(0, sy + 4 * font_height), cv::FONT_HERSHEY_COMPLEX, 0.5,
              cv::Scalar(10, 10, 110));

  // Draw line from face to info box
  cv::line(frame, cv::Point(box.x0, box.y0), cv::Point(ex, sy),
           cv::Scalar(10, 33, 10), 1);

}

bool FaceDetection::Start() {
  if (type_ == DETECT_LOOP) {
    return DetectLoop();
  } else if (type_ == DETECT_BY_IMAGE) {
    return DetectImage(FLAGS_test_image);
  } else if (type_ == DETECT_TAKE_PHOTO) {
    return TakePhoto();
  } else {
    LOG(FATAL) << "Invalid detect type:" << type_;
  }

  return true;
}

bool FaceDetection::DetectLoop() {
  cv::VideoCapture* capture = capture_->capture();
  if (capture == NULL) {
    LOG(WARNING) << "Invalid capture";
    return false;
  }

  face_index_.BuildIndexFromFaceDB();

  std::vector<face_box> faces;
  std::vector<double> dims;
  std::vector<uint64_t> face_ids;
  std::vector<double> face_dist;

  EmbeddingRequest request;
  EmbeddingReply reply;

  double t = 0;
  double fps;
  std::string fps_str;

  bool found = true;

	while (1) {
    t = (double)cv::getTickCount();

    cv::Mat frame;
		capture->read(frame);
    cv::Mat frame_tmp = frame;
		mtcnn_detect(sess_, graph_, frame, faces);
    LOG(INFO) << "faces size:" << faces.size();

		for (unsigned int i = 0; i < faces.size(); i++) {
			face_box& box = faces[i];
      cv::Mat face_img = NormalizedFaceRange(frame, box);
      DrawRectAndLandmark(frame, box);

      if (face_img.empty()) { continue; }
      std::string img_data;
      Mat2Base64(face_img, ".jpg", &img_data);
      if (img_data.empty()) { continue; }

      request.set_image_base64(img_data);
      reply = facenet_client_->GenerateEmbedding(request);
      if (reply.error() != 0) {
        LOG(WARNING) << "Generate embedding error";
        continue;
      }

      dims.clear();
      std::copy(reply.dim().begin(), reply.dim().end(), std::back_inserter(dims));

      // Nearst
      if (dims.size() == FaceIndex::kDims) {
        face_ids.clear();
        face_dist.clear();
        face_index_.GetNearst(3, dims, &face_ids, &face_dist);
        int min_face_id = -1;
        double min_face_dist = 100.0f;
        for (int i = 0; i < face_ids.size(); i++) {
          LOG(INFO) << "face id:" << face_ids[i];
          LOG(INFO) << "face dist:" << face_dist[i];
          if (min_face_dist > face_dist[i]) {
            min_face_dist = face_dist[i];
            min_face_id = face_ids[i];
          }
        }
        if (!face_ids.empty()) {
          LOG(INFO) << "min face id:" << min_face_id;
          LOG(INFO) << "min face dist:" << min_face_dist;
        }
        found = false;
        if (min_face_dist < FLAGS_min_face_dist) {
          FaceInfo face_info;
          if (face_index_.GetFaceInfo(min_face_id, &face_info)) {
            DrawFaceInfo(frame, face_info, box, i);
            found = true;
          }
        }
        if (!found) {
            cv::putText(frame, "NO RECORD", cv::Point(box.x0, box.y0 - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
          }
        LOG(INFO) << "--------------------------------";
      }
		}

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    fps = 1.0 / t;
    fps_str = "FPS:" + std::to_string(fps);
    // cv::FONT_HERSHEY_SIMPLEX
    cv::putText(frame, fps_str, cv::Point(0, 20), cv::FONT_HERSHEY_COMPLEX, 0.5,
                cv::Scalar(0, 255, 0));

    cv::imshow("FaceDetection", frame);
    // cv::Mat smooth_frame;
    // SmoothFace(frame, &smooth_frame);
		// cv::imshow("FaceDetection", smooth_frame);
    // cv::waitKey(10);
    cv::waitKey(1);

    faces.clear();
  }
  return true;
}

void FaceDetection::Stop() {}

void FaceDetection::Clean() {
	TF_Status* s = TF_NewStatus(); 
  if (sess_ != NULL) {
    TF_CloseSession(sess_, s);
    TF_DeleteSession(sess_, s);
    sess_ = NULL;
  }
  if (graph_ != NULL) {
    TF_DeleteGraph(graph_);
    TF_DeleteStatus(s);
    graph_ = NULL;
  }
}

bool FaceDetection::DetectImage(const std::string& image_file) {
  const cv::Size kNormalSize = cv::Size(160, 160);
	cv::Mat frame = cv::imread(image_file);
	if (!frame.data) {
		LOG(WARNING) << "Failed to read image file: " << image_file;
		return false;
	}

	std::vector<face_box> face_info;
	mtcnn_detect(sess_, graph_, frame, face_info);

	for (unsigned int i = 0; i < face_info.size(); i++) {
		face_box& box = face_info[i];
    cv::Mat face_img = frame(cv::Range(box.y0, box.y1), cv::Range(box.x0, box.x1));
    cv::imshow("face_" + std::to_string(i), face_img);

    cv::Mat affine_img;
    std::vector<cv::Point2f> landmarks;
		for (int l = 0; l < 5; l++) {
      landmarks.emplace_back(cv::Point2f(box.landmark.x[l], box.landmark.y[l]));
    }

    GetWarpAffineImage(face_img, landmarks, kNormalSize, &affine_img);
    cv::imshow("affine_" + std::to_string(i), affine_img);

    DrawRectAndLandmark(frame, box);
    cv::imshow("frame_" + std::to_string(i), frame);
	}

  return true;
}

int FaceDetection::GetWarpAffineImage(cv::Mat& src,
    const std::vector<cv::Point2f>& landmarks,
    const cv::Size& dst_size, cv::Mat* dst) {
  const cv::Point2f& left_eye = landmarks[0];
  const cv::Point2f& right_eye = landmarks[1];

  cv::Point2f eyes_center = cv::Point2f((left_eye.x + right_eye.x) * 0.5f,
                                        (left_eye.y + right_eye.y) * 0.5f);

  double dy = (right_eye.y - left_eye.y);
  double dx = (right_eye.x - left_eye.x);
  double angle = atan2(dy, dx) * 180.0 / CV_PI;

  cv::Mat mat = cv::getRotationMatrix2D(eyes_center, angle, 1.0);
  // cv::warpAffine(src, *dst, mat, dst_size);
  cv::warpAffine(src, *dst, mat, src.size());
  return 0;
}

#define JSON_STRING(_str_) rapidjson::StringRef(_str_.c_str())
void FaceDetection::WrapJson(const std::string& raw, std::string* data) {
  rapidjson::Document d;
  rapidjson::Document::AllocatorType& allocator = d.GetAllocator();
  d.SetObject();
  d.AddMember("data", JSON_STRING(raw), allocator);
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  d.Accept(writer);
  *data = std::move(buffer.GetString());
}

bool FaceDetection::TakePhoto() {
  std::string face_id_str;
  std::cout << "Enter id" << std::endl;
  std::cin >> face_id_str;
  if (face_id_str.empty()) {
    LOG(WARNING) << "Invalid face id";
    return false;
  }
  std::string face_img_file = face_id_str + ".jpg";
  uint64_t face_id = std::stoull(face_id_str);

  cv::VideoCapture* capture = capture_->capture();
  if (capture == NULL) {
    LOG(WARNING) << "Invalid capture";
    return false;
  }

  std::vector<face_box> faces;
  std::vector<double> dims;

  while (1) {
    cv::Mat frame;
    capture->read(frame);
    cv::Mat frame_tmp = frame;
    mtcnn_detect(sess_, graph_, frame, faces);
    if (faces.empty()) {
      LOG(WARNING) << "No faced detected";
      continue;
    }
    if (faces.size() > 1) {
      LOG(WARNING) << "Multi faced detected in mode DETECT_SINGLE_FACE";
      // continue;
    }

    face_box& box = faces[0];
    cv::Mat face_img = NormalizedFaceRange(frame, box);
    DrawRectAndLandmark(frame, box);

    int key = -1;
    if ((key = cv::waitKey(5)) != -1) {
      if (key == 'p') {
        cv::imwrite(face_img_file, face_img);
        LOG(INFO) << "Save face for face_id:" << face_id_str;
        cv::putText(frame, "Face Saved", cv::Point(0, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        if (GetFaceDims(face_img, &dims)) {
          if (!FaceDB::Instance().Put(face_id, dims)) {
            LOG(WARNING) << "Put face dims into face db error";
          } else {
            LOG(INFO) << "Put face dims into face db successfully";
          }
        }
        dims.clear();
      }
      if (key == 'q') {
        LOG(INFO) << "Exit take photo";
        return true;
      }
    }

    cv::imshow("TakePhoto", frame);
    cv::waitKey(1);
    faces.clear();
  }
  return true;
}

cv::Mat FaceDetection::NormalizedFaceRange(cv::Mat& frame, face_box& box) {
  if (box.x0 < 0) { box.x0 = 0; }
  if (box.x1 < 0) { box.x1 = 0; }
  if (box.y0 < 0) { box.y0 = 0; }
  if (box.y1 < 0) { box.y1 = 0; }
  if (box.x0 >= frame.size().width) { box.x0 = frame.size().width; }
  if (box.x1 >= frame.size().width) { box.x1 = frame.size().width; }
  if (box.y0 >= frame.size().height) { box.y0 = frame.size().height; }
  if (box.y1 >= frame.size().height) { box.y1 = frame.size().height; }
  return frame(cv::Rect(box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0));
}

bool FaceDetection::GetFaceDims(cv::Mat& face_img, std::vector<double>* dims) {
  if (face_img.empty()) { return false; }
  std::string img_data;
  Mat2Base64(face_img, ".jpg", &img_data);
  if (img_data.empty()) { return false; }

  EmbeddingRequest request;
  EmbeddingReply reply;

  request.set_image_base64(img_data);
  reply = facenet_client_->GenerateEmbedding(request);
  if (reply.error() != 0) {
    LOG(WARNING) << "Generate embedding error";
    return false;
  }
  std::copy(reply.dim().begin(), reply.dim().end(), std::back_inserter(*dims));
  return true;
}

