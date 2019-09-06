#include "util/opencv_util.h"

void DrawText(cv::Mat& frame, std::string text) {
  int font_face = cv::FONT_HERSHEY_COMPLEX;
  double font_scale = 2;
  int thickness = 2;
  int baseline;
  cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);

  cv::Point origin;
  origin.x = frame.cols / 2 - text_size.width / 2;
  origin.y = frame.rows / 2 + text_size.height / 2;
  cv::putText(frame, text, origin, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 8, 0);
}

void SmoothFace(cv::Mat& src, cv::Mat* res) {
  cv::Mat tmp;
	int bilateral_filter = 30;
	WhiteFace(src, 1.1, 68);
	GaussianBlur(src, src, cv::Size(9, 9), 0, 0);
	bilateralFilter(src, tmp, bilateral_filter, bilateral_filter * 2,
                  bilateral_filter / 2);
	cv::GaussianBlur(tmp, *res, cv::Size(0, 0), 9);
	cv::addWeighted(tmp, 1.5, *res, -0.5, 0, *res);
}

void WhiteFace(cv::Mat& src, double alpha, int beta) {
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			for (int c = 0; c < 3; c++) {
				src.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(
            alpha * (src.at<cv::Vec3b>(y, x)[c]) + beta);
			}
		}
	}
}
