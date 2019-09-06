#ifndef _TENSORFLOW_MTCNN_H_
#define _TENSORFLOW_MTCNN_H_

#include "tensorflow/c/c_api.h"
#include <opencv2/opencv.hpp>
#include "mtcnn.h"

void mtcnn_detect(TF_Session* sess, TF_Graph* graph, cv::Mat& img,
                  std::vector<face_box>& face_list);

TF_Session* load_graph(const char* frozen_fname, TF_Graph** p_graph);

#endif
