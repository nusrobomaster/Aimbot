#include "yolo_0526_trt.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>

#include <cuda_runtime_api.h>
#include <yaml-cpp/yaml.h>

#include "tools/logger.hpp"

namespace auto_aim
{

namespace
{
class TrtLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char* msg) noexcept override
  {
    if (severity <= Severity::kWARNING)
      tools::logger()->warn("[TensorRT] {}", msg);
  }
};

// Portable float -> FP16 (uint16_t) for host-side input fill
inline uint16_t float_to_fp16(float f)
{
  uint32_t u;
  std::memcpy(&u, &f, sizeof(float));
  const uint32_t sign = (u >> 16) & 0x8000u;
  const uint32_t rest = u & 0x7fffffffu;
  if (rest >= 0x47800000u)  // inf or nan
    return sign | 0x7c00u;
  if (rest < 0x38800000u) {  // subnormal
    uint32_t frac = rest | 0x3f800000u;
    return sign | (frac >> 13);
  }
  return sign | ((rest + 0xc8000000u) >> 13);
}
}  // namespace

YOLO_0526_TRT::YOLO_0526_TRT(const std::string& config_path, bool debug)
  : debug_(debug)
{
  auto yaml = YAML::LoadFile(config_path);
  model_path_   = yaml["yolo_engine_path"].as<std::string>();
  conf_thresh_  = yaml["yolo_conf_threshold"].as<float>();

  TrtLogger trt_logger;
  runtime_ = nvinfer1::createInferRuntime(trt_logger);
  if (!runtime_) {
    tools::logger()->error("[YOLO_0526_TRT] createInferRuntime failed");
    return;
  }

  std::ifstream in(model_path_, std::ios::binary);
  if (!in.good()) {
    tools::logger()->error("[YOLO_0526_TRT] Cannot open engine file: {}", model_path_);
    return;
  }
  in.seekg(0, std::ios::end);
  const size_t size = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<char> engine_buf(size);
  in.read(engine_buf.data(), size);
  in.close();

  engine_ = runtime_->deserializeCudaEngine(engine_buf.data(), size);
  if (!engine_) {
    tools::logger()->error("[YOLO_0526_TRT] deserializeCudaEngine failed");
    return;
  }

  context_ = engine_->createExecutionContext();
  if (!context_) {
    tools::logger()->error("[YOLO_0526_TRT] createExecutionContext failed");
    return;
  }

  const int nb = engine_->getNbIOTensors();
  for (int i = 0; i < nb; ++i) {
    const char* name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
      input_index_ = i;
    else
      output_index_ = i;
  }

  if (input_index_ < 0 || output_index_ < 0) {
    tools::logger()->error("[YOLO_0526_TRT] Could not find input/output tensor indices");
    return;
  }

  nvinfer1::Dims input_dims  = context_->getTensorShape(engine_->getIOTensorName(input_index_));
  nvinfer1::Dims output_dims = context_->getTensorShape(engine_->getIOTensorName(output_index_));

  // Input: (1, 3, 640, 640) FP16; use fixed size if dynamic (-1) dims
  size_t input_elems = 1;
  for (int j = 0; j < input_dims.nbDims; ++j)
    input_elems *= (input_dims.d[j] > 0 ? input_dims.d[j] : (j == 0 ? 1 : (j == 1 ? 3 : 640)));
  input_byte_ = input_elems * sizeof(uint16_t);

  // Output: (1, 25200, 23) float32
  size_t output_elems = 1;
  for (int j = 0; j < output_dims.nbDims; ++j)
    output_elems *= (output_dims.d[j] > 0 ? output_dims.d[j] : (j == 0 ? 1 : (j == 1 ? 25200 : 23)));
  output_byte_ = output_elems * sizeof(float);
  h_output_.resize(output_elems);

  if (cudaMalloc(&d_input_, input_byte_) != cudaSuccess ||
      cudaMalloc(&d_output_, output_byte_) != cudaSuccess) {
    tools::logger()->error("[YOLO_0526_TRT] cudaMalloc failed");
    return;
  }

  input_blob_.resize(3 * INPUT_H * INPUT_W);
  letterbox_buf_.create(cv::Size(INPUT_W, INPUT_H), CV_8UC3);
  rgb_buf_.create(cv::Size(INPUT_W, INPUT_H), CV_8UC3);

  tools::logger()->info("[YOLO_0526_TRT] Engine loaded: {} (input {} bytes, output {} elems)",
                        model_path_, input_byte_, output_elems);
}

YOLO_0526_TRT::~YOLO_0526_TRT()
{
  if (d_input_)  { cudaFree(d_input_);  d_input_  = nullptr; }
  if (d_output_) { cudaFree(d_output_); d_output_ = nullptr; }
  if (context_)  { delete context_;  context_  = nullptr; }
  if (engine_)   { delete engine_;   engine_   = nullptr; }
  if (runtime_)  { delete runtime_;  runtime_  = nullptr; }
}

void YOLO_0526_TRT::letterbox(
  const cv::Mat& img, cv::Mat& out,
  float& scale, int& pad_w, int& pad_h) const
{
  const int w = img.cols;
  const int h = img.rows;
  const float r = std::min(
    static_cast<float>(INPUT_W) / static_cast<float>(w),
    static_cast<float>(INPUT_H) / static_cast<float>(h));

  const int new_w = static_cast<int>(std::round(w * r));
  const int new_h = static_cast<int>(std::round(h * r));

  pad_w = (INPUT_W - new_w) / 2;
  pad_h = (INPUT_H - new_h) / 2;
  scale = r;

  cv::Mat resized;
  cv::resize(img, resized, cv::Size(new_w, new_h));

  if (out.cols != INPUT_W || out.rows != INPUT_H || out.type() != CV_8UC3)
    out.create(cv::Size(INPUT_W, INPUT_H), CV_8UC3);
  out.setTo(cv::Scalar(114, 114, 114));
  resized.copyTo(out(cv::Rect(pad_w, pad_h, new_w, new_h)));
}

std::list<Armor> YOLO_0526_TRT::detect(const cv::Mat& raw_img, int /*frame_count*/)
{
  if (raw_img.empty()) {
    tools::logger()->warn("[YOLO_0526_TRT] Empty image!");
    return {};
  }
  if (!context_ || !d_input_ || !d_output_)
    return {};

  float scale = 1.0f;
  int pad_w = 0, pad_h = 0;
  letterbox(raw_img, letterbox_buf_, scale, pad_w, pad_h);
  cv::cvtColor(letterbox_buf_, rgb_buf_, cv::COLOR_BGR2RGB);

  const int c_step = INPUT_H * INPUT_W;
  for (int y = 0; y < INPUT_H; ++y) {
    const auto* row_ptr = rgb_buf_.ptr<cv::Vec3b>(y);
    for (int x = 0; x < INPUT_W; ++x) {
      const cv::Vec3b& p = row_ptr[x];
      input_blob_[0 * c_step + y * INPUT_W + x] = float_to_fp16(static_cast<float>(p[0]) / 255.0f);
      input_blob_[1 * c_step + y * INPUT_W + x] = float_to_fp16(static_cast<float>(p[1]) / 255.0f);
      input_blob_[2 * c_step + y * INPUT_W + x] = float_to_fp16(static_cast<float>(p[2]) / 255.0f);
    }
  }

  if (cudaMemcpy(d_input_, input_blob_.data(), input_byte_, cudaMemcpyHostToDevice) != cudaSuccess) {
    tools::logger()->warn("[YOLO_0526_TRT] cudaMemcpy H2D failed");
    return {};
  }

  const int nb = engine_->getNbIOTensors();
  std::vector<void*> bindings(nb);
  bindings[input_index_]  = d_input_;
  bindings[output_index_] = d_output_;
  if (!context_->executeV2(bindings.data())) {
    tools::logger()->warn("[YOLO_0526_TRT] executeV2 failed");
    return {};
  }

  if (cudaMemcpy(h_output_.data(), d_output_, output_byte_, cudaMemcpyDeviceToHost) != cudaSuccess) {
    tools::logger()->warn("[YOLO_0526_TRT] cudaMemcpy D2H failed");
    return {};
  }

  const int stride = 22;
  const size_t num_rows = (output_byte_ / sizeof(float)) / stride;
  return decode(h_output_.data(), num_rows, conf_thresh_, scale, pad_w, pad_h, raw_img.size());
}

std::list<Armor> YOLO_0526_TRT::decode(
  const float* output, size_t num_rows,
  float conf_thresh,
  float scale, int pad_w, int pad_h,
  const cv::Size& orig_size) const
{
  const int stride = 22;

  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  std::vector<int> color_ids;
  std::vector<int> digit_ids;
  std::vector<std::vector<cv::Point2f>> all_keypoints;

  const int img_w = orig_size.width;
  const int img_h = orig_size.height;

  boxes.reserve(num_rows);
  confidences.reserve(num_rows);
  color_ids.reserve(num_rows);
  digit_ids.reserve(num_rows);
  all_keypoints.reserve(num_rows);

  for (size_t i = 0; i < num_rows; ++i) {
    const float* row = output + i * stride;
    const float conf = row[8];
    if (conf < conf_thresh)
      continue;

    std::vector<cv::Point2f> kps_raw(4);
    for (int k = 0; k < 4; ++k) {
      float px = (row[2 * k + 0] - static_cast<float>(pad_w)) / scale;
      float py = (row[2 * k + 1] - static_cast<float>(pad_h)) / scale;
      px = std::clamp(px, 0.0f, static_cast<float>(img_w - 1));
      py = std::clamp(py, 0.0f, static_cast<float>(img_h - 1));
      kps_raw[k] = cv::Point2f(px, py);
    }
    std::vector<cv::Point2f> kps = {kps_raw[0], kps_raw[3], kps_raw[2], kps_raw[1]};
    float min_x = std::min({kps[0].x, kps[1].x, kps[2].x, kps[3].x});
    float min_y = std::min({kps[0].y, kps[1].y, kps[2].y, kps[3].y});
    float max_x = std::max({kps[0].x, kps[1].x, kps[2].x, kps[3].x});
    float max_y = std::max({kps[0].y, kps[1].y, kps[2].y, kps[3].y});

    if (max_x <= min_x || max_y <= min_y)
      continue;

    int best_color = 0;
    float best_color_score = row[9];
    for (int c = 1; c < 4; ++c) {
      float s = row[9 + c];
      if (s > best_color_score) { best_color_score = s; best_color = c; }
    }

    int best_digit = 0;
    float best_digit_score = row[13];
    for (int d = 1; d < 9; ++d) {
      float s = row[13 + d];
      if (s > best_digit_score) { best_digit_score = s; best_digit = d; }
    }

    if (best_digit < 0 || best_digit > 8) best_digit = 0;
    if (best_color < 0 || best_color > 3) best_color = 0;

    const int x = static_cast<int>(std::round(min_x));
    const int y = static_cast<int>(std::round(min_y));
    const int w = static_cast<int>(std::round(max_x - min_x));
    const int h = static_cast<int>(std::round(max_y - min_y));

    boxes.emplace_back(x, y, w, h);
    confidences.emplace_back(conf);
    color_ids.emplace_back(best_color);
    digit_ids.emplace_back(best_digit);
    all_keypoints.emplace_back(std::move(kps));
  }

  std::vector<int> nms_indices;
  cv::dnn::NMSBoxes(boxes, confidences, conf_thresh, 0.3f, nms_indices);

  std::list<Armor> armors;
  for (int i : nms_indices)
    armors.emplace_back(color_ids[i], digit_ids[i], confidences[i], boxes[i], all_keypoints[i]);
  return armors;
}

}  // namespace auto_aim
