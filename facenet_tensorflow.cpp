// File   facenet_tensorflow.cpp
// Author lidongming1@360.cn
// Date   2019-09-20 17:44:54
// Brief

#include <fstream>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include "facenet_tensorflow.h"

using namespace tensorflow;

// Init tensorflow session and create graph
int FaceNetTensorflow::Init(const std::string& model_file) {
  // Initialize tensorflow session
  Status status = NewSession(SessionOptions(), &session_);
  if (!status.ok()) {
		LOG(FATAL) << "Init tensorflow session failed:" << status.ToString();
    return -1;
  } else {
		LOG(INFO) << "Init tensorflow session successfully";
  }

  // Load graph
  GraphDef graph_def;
  status = ReadBinaryProto(Env::Default(), model_file, &graph_def);
  if (!status.ok()) {
		LOG(FATAL) << "Load model " << model_file << " failed:" << status.ToString();
		return -1;
	} else {
		LOG(INFO) << "Load model " << model_file << " successfully";
  }

  // Add graph to the session
  status = session_->Create(graph_def);
  if (!status.ok()) {
		LOG(FATAL) << "Create graph in the session failed:" << status.ToString();
		return -1;
	} else {
		LOG(INFO) << "Create graph in the session successfully";
  }
  return 0;
}

int FaceNetTensorflow::GenerateEmbedding(std::vector<cv::Mat>& faces,
    std::vector<std::vector<double>>* face_dims) {
  int faces_size = faces.size();
  Tensor input_tensor(DT_FLOAT, TensorShape({faces_size, 160, 160, 3}));
  float* input_tensor_data = input_tensor.flat<float>().data();

  for (int i = 0; i < faces_size; i++) {
    cv::Mat& img = faces[i];
    cvtColor(img, img, cv::COLOR_RGB2BGR);

    // mean and std
    cv::Mat temp = img.reshape(1, img.rows * 3);
    cv::Mat mean3;
    cv::Mat stddev3;
    cv::meanStdDev(temp, mean3, stddev3);
    double mean_pxl = mean3.at<double>(0);
    double stddev_pxl = stddev3.at<double>(0);

    cv::Mat img2;
    img.convertTo(img2, CV_64FC1);
    img = img2;
    cv::Mat mat(4, 1, CV_64FC1);
    mat.at<double>(0, 0) = mean_pxl;
    mat.at<double>(1, 0) = mean_pxl;
    mat.at<double>(2, 0) = mean_pxl;
    mat.at<double>(3, 0) = 0;
    img = img - mat;
    img = img / stddev_pxl;

    cv::Mat normalize_image(160, 160, CV_32FC3, input_tensor_data + i * 160 * 160 * 3);
    img.convertTo(normalize_image, CV_32FC3);
  }

  Tensor phase_tensor(tensorflow::DT_BOOL, TensorShape());
  phase_tensor.scalar<bool>()() = false;

  std::vector<std::pair<std::string, Tensor>> inputs = {
    { "input:0", input_tensor },
    { "phase_train:0", phase_tensor }
  };

  std::vector<string> output_names = { "embeddings:0" };
  std::vector<Tensor> outputs;

  Status status = session_->Run(inputs, output_names, {}, &outputs);
  if (!status.ok()) {
    std::cerr << "get embeddings error:" << status.ToString() << std::endl;
    return -1;
  }

  float* output_tensor_data = outputs[0].flat<float>().data();
  cv::Mat output_mat;
  for (int i = 0; i < faces_size; i++) {
    std::vector<double> dims;
    for (int j = 0; j < 512; j++) {
      dims.push_back(output_tensor_data[j + i * 512]);
    }
    face_dims->emplace_back(std::move(dims));
  }

  return 0;
}
