#include <fstream>
#include <utility>
#include "tf_mtcnn.h"

const int NMS_UNION = 1;
const int NMS_MIN = 2;

const float kAlpha = 0.0078125;
const float kMean = 127.5;
const float kPNetThreshold = 0.6;
const float kRNetThreshold = 0.7;
const float kOnetThreshold = 0.9;
const	int kChannel = 3;
const	int kStride = 2;
const	int kCellSize = 12;
const int kMinSize = 40;
const float kFactor = 0.709;

int TFMtcnn::Init(const std::string& model_file) {
	std::vector<char> model_buf;
	std::ifstream fs(model_file, std::ios::binary | std::ios::in);
	if (!fs.good()) {
		std::cerr << "Load model error:" << model_file << std::endl;
		return -1;
	}
	fs.seekg(0, std::ios::end);
	int size = fs.tellg();
	fs.seekg(0, std::ios::beg);
	model_buf.resize(size);
	fs.read(model_buf.data(), size);
	fs.close();

	TF_Status* s = TF_NewStatus();
	TF_Buffer graph_def = {model_buf.data(), model_buf.size(), nullptr};
	graph_ = TF_NewGraph();
	TF_ImportGraphDefOptions* import_opts = TF_NewImportGraphDefOptions();
	TF_ImportGraphDefOptionsSetPrefix(import_opts, "");
	TF_GraphImportGraphDef(graph_, &graph_def, import_opts, s);
	if (TF_GetCode(s) != TF_OK) {
    std::cerr << "Load graph_ error:" << TF_Message(s) << std::endl;
		return -1;
	}

	TF_SessionOptions* session_opts = TF_NewSessionOptions();
	session_ = TF_NewSession(graph_, session_opts, s);

	TF_DeleteStatus(s);
  return 0;
}

void TFMtcnn::GenerateBoundingBox(const float* confidence_data,
                                  const float* reg_data, float scale,
                                  float threshold, int h, int w, bool transposed,
                                  std::vector<FaceBox>* output) {
  float score = 0.0f;
  float top_x = 0.0f;
  float top_y = 0.0f;
  float bottom_x = 0.0f;
  float bottom_y = 0.0f;
  int score_offset = 0;
  int reg_offset = 0;

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
      score_offset = 2 * w * y + 2 * x + 1;
			score = confidence_data[score_offset];
			if (score >= threshold) {
				top_x = (int)((x * kStride + 1) / scale);
				top_y = (int)((y * kStride + 1) / scale);
				bottom_x = (int)((x * kStride + kCellSize) / scale);
				bottom_y = (int)((y * kStride + kCellSize) / scale);

				FaceBox box;
				box.x0 = top_x;
				box.y0 = top_y;
				box.x1 = bottom_x;
				box.y1 = bottom_y;
				box.score = score;

				reg_offset = (w * 4) * y + 4 * x;
				if (transposed) {
					box.regress[1] = reg_data[reg_offset];
					box.regress[0] = reg_data[reg_offset + 1]; 
					box.regress[3] = reg_data[reg_offset + 2];
					box.regress[2] = reg_data[reg_offset + 3];
				} else {
					box.regress[0] = reg_data[reg_offset];
					box.regress[1] = reg_data[reg_offset + 1]; 
					box.regress[2] = reg_data[reg_offset + 2];
					box.regress[3] = reg_data[reg_offset + 3];
				}
				output->emplace_back(box);
			}
		}
  }
}

static void dummy_deallocator(void* data, size_t len, void* arg) {}

void TFMtcnn::PNet(cv::Mat& img, const ScaleWindow& win,
                   std::vector<FaceBox>* boxes) {
	cv::Mat resized;
	int scale_h = win.h;
	int scale_w = win.w;
	float scale = win.scale;

	cv::resize(img, resized, cv::Size(scale_w, scale_h), 0, 0);

	std::vector<TF_Output> input_names;
	std::vector<TF_Tensor*> input_values;

	TF_Operation* input_name = TF_GraphOperationByName(graph_, "pnet/input");

	input_names.push_back({input_name, 0});

	const int64_t dim[4] = {1, scale_h, scale_w, 3};

	TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, dim, 4, resized.ptr(),
                                         sizeof(float) * scale_w * scale_h * 3,
                                         dummy_deallocator, nullptr);

	input_values.push_back(input_tensor);

	std::vector<TF_Output> output_names;

	TF_Operation* output_name = TF_GraphOperationByName(graph_, "pnet/conv4-2/BiasAdd");
	output_names.push_back({output_name, 0});

	output_name = TF_GraphOperationByName(graph_, "pnet/prob1");
	output_names.push_back({output_name, 0});

	std::vector<TF_Tensor*> output_values(output_names.size(), nullptr);

	TF_Status* s= TF_NewStatus();
	TF_SessionRun(session_, nullptr, input_names.data(), input_values.data(),
                input_names.size(), output_names.data(), output_values.data(),
                output_names.size(), nullptr, 0, nullptr, s);

	const float* conf_data=(const float*)TF_TensorData(output_values[1]);
	const float* reg_data=(const float*)TF_TensorData(output_values[0]);

	int feature_h = TF_Dim(output_values[0], 1);
	int feature_w = TF_Dim(output_values[0], 2);
	// int conf_size = feature_h * feature_w * 2;

	std::vector<FaceBox> candidate_boxes;
	GenerateBoundingBox(conf_data, reg_data, scale, kPNetThreshold,
                      feature_h, feature_w, true, &candidate_boxes);
	Normalize(candidate_boxes, 0.5, NMS_UNION, boxes);

	TF_DeleteStatus(s);
	TF_DeleteTensor(output_values[0]);
	TF_DeleteTensor(output_values[1]);
	TF_DeleteTensor(input_tensor);
}

void TFMtcnn::RNet(cv::Mat& img, std::vector<FaceBox>& pnet_boxes,
                   std::vector<FaceBox>* output_boxes) {
	int batch = pnet_boxes.size();
	int height = 24;
	int width = 24;

	int input_size = batch * height * width * kChannel;
	std::vector<float> input_buffer(input_size);
	float* input_data = input_buffer.data();

  int patch_size = width * height * kChannel;
	for (int i = 0; i < batch; i++) {
		CopyPatch(img, pnet_boxes[i], input_data, height, width);
		input_data += patch_size;
	}

	std::vector<TF_Output> input_names;

	TF_Operation* input_name = TF_GraphOperationByName(graph_, "rnet/input");
	input_names.push_back({input_name, 0});

	std::vector<TF_Tensor*> input_values;
	const int64_t dim[4] = {batch, height, width, kChannel};
	TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, dim, 4, input_buffer.data(),
                                         sizeof(float) * input_size,
                                         dummy_deallocator, nullptr);
	input_values.push_back(input_tensor);

	std::vector<TF_Output> output_names;
	TF_Operation* output_name = TF_GraphOperationByName(graph_,"rnet/conv5-2/conv5-2");
	output_names.push_back({output_name, 0});
	output_name = TF_GraphOperationByName(graph_,"rnet/prob1");
	output_names.push_back({output_name,0});

	std::vector<TF_Tensor*> output_values(output_names.size(), nullptr);
	TF_Status* s = TF_NewStatus();
	TF_SessionRun(session_, nullptr, input_names.data(), input_values.data(),
                input_names.size(), output_names.data(), output_values.data(),
                output_names.size(), nullptr, 0, nullptr, s);

	const float* conf_data=(const float *)TF_TensorData(output_values[1]);
	const float* reg_data=(const float *)TF_TensorData(output_values[0]);

	for (int i = 0; i < batch; i++) {
		if (conf_data[1] > kRNetThreshold) {
			FaceBox output_box;
			FaceBox& input_box=pnet_boxes[i];
			output_box.x0=input_box.x0;
			output_box.y0=input_box.y0;
			output_box.x1=input_box.x1;
			output_box.y1=input_box.y1;
			output_box.score = *(conf_data+1);
			output_box.regress[0]=reg_data[1];
			output_box.regress[1]=reg_data[0];
			output_box.regress[2]=reg_data[3];
			output_box.regress[3]=reg_data[2];
			output_boxes->emplace_back(output_box);
		}
		conf_data += 2;
		reg_data += 4;
	}

	TF_DeleteStatus(s);
	TF_DeleteTensor(output_values[0]);
	TF_DeleteTensor(output_values[1]);
	TF_DeleteTensor(input_tensor);
}

void TFMtcnn::ONet(cv::Mat& img, std::vector<FaceBox>& rnet_boxes,
                   std::vector<FaceBox>* output_boxes) {
	int batch = rnet_boxes.size();
	int height = 48;
	int width = 48;
	int input_size = batch * height * width * kChannel;

	std::vector<float> input_buffer(input_size);
	float* input_data = input_buffer.data();

	for (int i = 0; i < batch; i++) {
		int patch_size = width * height * kChannel;
		CopyPatch(img, rnet_boxes[i], input_data, height, width);
		input_data += patch_size;
	}

	std::vector<TF_Output> input_names;
	std::vector<TF_Tensor*> input_values;

	TF_Operation* input_name = TF_GraphOperationByName(graph_, "onet/input");
	input_names.push_back({input_name, 0});

	const int64_t dim[4] = { batch, height, width, kChannel };
	TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, dim, 4, input_buffer.data(),
                                         sizeof(float) * input_size,
                                         dummy_deallocator, nullptr);
	input_values.push_back(input_tensor);

	std::vector<TF_Output> output_names;
	TF_Operation* output_name = TF_GraphOperationByName(graph_,"onet/conv6-2/conv6-2");
	output_names.push_back({output_name,0});
	output_name = TF_GraphOperationByName(graph_,"onet/conv6-3/conv6-3");
	output_names.push_back({output_name,0});
	output_name = TF_GraphOperationByName(graph_,"onet/prob1");
	output_names.push_back({output_name,0});

	std::vector<TF_Tensor*> output_values(output_names.size(), nullptr);
	TF_Status* s = TF_NewStatus();
	TF_SessionRun(session_, nullptr, input_names.data(), input_values.data(),
                input_names.size(), output_names.data(), output_values.data(),
                output_names.size(), nullptr, 0, nullptr, s);

	const float* conf_data = (const float*)TF_TensorData(output_values[2]);
	const float* reg_data = (const float*)TF_TensorData(output_values[0]);
	const float* points_data = (const float*)TF_TensorData(output_values[1]);

	for (int i = 0; i < batch; i++) {
		if (conf_data[1] > kOnetThreshold) {
			FaceBox output_box;
			FaceBox& input_box=rnet_boxes[i];
			output_box.x0=input_box.x0;
			output_box.y0=input_box.y0;
			output_box.x1=input_box.x1;
			output_box.y1=input_box.y1;
			output_box.score = conf_data[1];
			output_box.regress[0]=reg_data[1];
			output_box.regress[1]=reg_data[0];
			output_box.regress[2]=reg_data[3];
			output_box.regress[3]=reg_data[2];
			for (int j = 0; j < 5; j++) {
				output_box.landmark.x[j] = *(points_data + j + 5);
				output_box.landmark.y[j] = *(points_data + j);
			}
			output_boxes->emplace_back(output_box);
		}
		conf_data += 2;
		reg_data += 4;
		points_data += 10;
	}

	TF_DeleteStatus(s);
	TF_DeleteTensor(output_values[0]);
	TF_DeleteTensor(output_values[1]);
	TF_DeleteTensor(output_values[2]);
	TF_DeleteTensor(input_tensor);
}

void TFMtcnn::Detect(cv::Mat& src, std::vector<FaceBox>* face_boxes) {
	cv::Mat img;
	src.convertTo(img, CV_32FC3);
	img = (img - kMean) * kAlpha;
	img = img.t();
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

	int img_h = img.rows;
	int img_w = img.cols;
	std::vector<ScaleWindow> scale_windows;
	PyramidScales(img_h, img_w, kMinSize, kFactor, &scale_windows);

	std::vector<FaceBox> pnet_boxes;
	std::vector<FaceBox> pnet_boxes_final;
  for (const ScaleWindow& scale_window : scale_windows) {
		std::vector<FaceBox> boxes;
		PNet(img, scale_window, &boxes);
		pnet_boxes.insert(pnet_boxes.end(), boxes.begin(), boxes.end());
	}
	ProcessBoxes(pnet_boxes, img_h, img_w, &pnet_boxes_final);

	std::vector<FaceBox> rnet_boxes;
	std::vector<FaceBox> rnet_boxes_final;
	RNet(img, pnet_boxes_final, &rnet_boxes);
	ProcessBoxes(rnet_boxes, img_h, img_w, &rnet_boxes_final);

  float h = 0.0f;
  float w = 0.0f;
	std::vector<FaceBox> onet_boxes;
	ONet(img, rnet_boxes_final, &onet_boxes);
  for (FaceBox& box : onet_boxes) {
		h = box.x1 - box.x0 + 1;
		w = box.y1 - box.y0 + 1;
		for (int j = 0; j < 5; j++) {
			box.landmark.x[j] = box.x0 + w * box.landmark.x[j] - 1;
			box.landmark.y[j] = box.y0 + h * box.landmark.y[j] - 1;
		}
	}

	Regress(&onet_boxes);
	Normalize(onet_boxes, 0.7, NMS_MIN, face_boxes);

  for (FaceBox& box : *face_boxes) {
		std::swap(box.x0, box.y0);
		std::swap(box.x1 ,box.y1);

		for(int l = 0; l < 5; l++) {
			std::swap(box.landmark.x[l], box.landmark.y[l]);
		}
	}
}

void TFMtcnn::PyramidScales(int height, int width, int min_size, float factor,
                            std::vector<ScaleWindow>* scale_windows) {
	int min_side = std::min(height, width);
	double m = 12.0 / min_size;
	min_side = min_side * m;
	double cur_scale = 1.0;
	double scale;

	while (min_side >= 12) {
		scale = m * cur_scale;
		cur_scale = cur_scale * factor; 
		min_side *= factor;

		int hs = std::ceil(height * scale);
		int ws = std::ceil(width * scale);

    ScaleWindow s;
    s.h = hs;
    s.w = ws;
    s.scale = scale;
		scale_windows->emplace_back(s);
	}
}

void TFMtcnn::ProcessBoxes(std::vector<FaceBox>& input, int img_h, int img_w,
                           std::vector<FaceBox>* res) {
	Normalize(input, 0.7, NMS_UNION, res); 
	Regress(res);
	Square(res);
	Padding(img_h, img_w, res);
} 

void TFMtcnn::Normalize(std::vector<FaceBox>& input, float threshold, int type,
                        std::vector<FaceBox>* output) {
	std::sort(input.begin(), input.end(),
			[](const FaceBox& a, const FaceBox& b) {
			  return a.score > b.score;  
			});

	int box_num = input.size();
	std::vector<int> merged(box_num, 0);
  float area0 = 0.0f;
  float area1 = 0.0f;
  float score = 0.0f;

	for (int i = 0; i < box_num; i++) { 
		if (merged[i]) { continue; }

    const FaceBox& c = input[i];
		output->emplace_back(c);
		area0 = (c.y1 - c.y0 + 1) * (c.x1 - c.x0 + 1);

		for(int j = i + 1; j < box_num; j++) {
			if (merged[j]) { continue; }

      const FaceBox& n = input[j];

			float inner_x0 = std::max(c.x0, n.x0);
			float inner_y0 = std::max(c.y0, n.y0);
			float inner_x1 = std::min(c.x1, n.x1);
			float inner_y1 = std::min(c.y1, n.y1);
			float inner_h = inner_y1 - inner_y0 + 1;
			float inner_w = inner_x1 - inner_x0 + 1;
			if (inner_h <= 0 || inner_w <= 0) { continue; }

			float inner_area = inner_h * inner_w;
			area1 = (n.y1 - n.y0 + 1) * (n.x1 - n.x0 + 1);

			if (type == NMS_UNION) {
				score = inner_area / (area0 + area1 - inner_area);
      } else if (type == NMS_MIN) {
				score = inner_area / std::min(area0, area1);
			} else {
				score = 0.0f;
			}

			if (score > threshold) {
				merged[j] = 1;
      }
		}
	}
}

void TFMtcnn::Regress(std::vector<FaceBox>* boxes) {
  float h = 0.0f;
  float w = 0.0f;
  for (FaceBox& box : *boxes) {
		h = box.y1 - box.y0 + 1;
		w = box.x1 - box.x0 + 1;
		box.x0 = box.x0 + w * box.regress[0];
		box.y0 = box.y0 + h * box.regress[1];
		box.x1 = box.x1 + w * box.regress[2];
		box.y1 = box.y1 + h * box.regress[3];
	}    
}

void TFMtcnn::Square(std::vector<FaceBox>* boxes) {
  float h = 0.0f;
  float w = 0.0f;
  float l = 0.0f;
  for (FaceBox& box : *boxes) {
		h = box.y1 - box.y0 + 1;
		w = box.x1 - box.x0 + 1;
		l = std::max(h, w);
		box.x0 = box.x0 + (w - l) * 0.5;
		box.y0 = box.y0 + (h - l) * 0.5;
		box.x1 = box.x0 + l - 1;
		box.y1 = box.y0 + l - 1;
	}
}

void TFMtcnn::Padding(int img_h, int img_w, std::vector<FaceBox>* boxes) {
  for (FaceBox& box : *boxes) {
		box.px0 = std::max(box.x0, 1.0f);
		box.py0 = std::max(box.y0, 1.0f);
		box.px1 = std::min(box.x1, (float)img_w);
		box.py1 = std::min(box.y1, (float)img_h);
	}
} 

void TFMtcnn::CopyPatch(const cv::Mat& img, FaceBox& box, float* data_to,
                        int height, int width) {
	cv::Mat resized(height, width, CV_32FC3, data_to);
	cv::Mat chop_img = img(cv::Range(box.py0, box.py1), cv::Range(box.px0, box.px1));
	int pad_top    = std::abs(box.py0 - box.y0);
	int pad_left   = std::abs(box.px0 - box.x0);
	int pad_bottom = std::abs(box.py1 - box.y1);
	int pad_right  = std::abs(box.px1 - box.x1);
	cv::copyMakeBorder(chop_img, chop_img, pad_top, pad_bottom, pad_left,
                     pad_right, cv::BORDER_CONSTANT, cv::Scalar(0));
	cv::resize(chop_img, resized, cv::Size(width, height), 0, 0);
}
