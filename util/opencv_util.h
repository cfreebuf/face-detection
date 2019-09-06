#ifndef _OPENCV_UTIL_H
#define _OPENCV_UTIL_H

#include <opencv2/imgproc.hpp>
void DrawText(cv::Mat& frame, std::string text);

void SmoothFace(cv::Mat& src, cv::Mat* res);
void WhiteFace(cv::Mat& src, double alpha, int beta);

#endif
