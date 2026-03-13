#include "yolo_0526_ort.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <yaml-cpp/yaml.h>

#include "tools/logger.hpp"

namespace auto_aim
{

YOLO_0526_ORT::YOLO_0526_ORT(const std::string & config_path, bool debug)
: debug_(debug),
  env_(ORT_LOGGING_LEVEL_WARNING, "yolo_0526"),
  session_options_()
{
  auto yaml = YAML::LoadFile(config_path);

  model_path_ = yaml["yolo_onnx_model_path"].as<std::string>();
  conf_thresh_ = yaml["yolo_conf_threshold"].as<float>();

  // Enable high level optimizations
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Try to enable TensorRT EP if built with it
#if defined(ORT_TENSORRT_EXECUTION_PROVIDER)
  {
    OrtTensorRTProviderOptionsV2 trt_options;
    std::memset(&trt_options, 0, sizeof(trt_options));
    trt_options.trt_fp16_enable = 1;  // Prefer FP16 on Jetson
    Ort::ThrowOnError(
      OrtSessionOptionsAppendExecutionProvider_TensorRT_V2(session_options_, &trt_options));
  }
#endif

  // Try to enable CUDA EP if available
#if defined(ORT_CUDA_EXECUTION_PROVIDER)
  {
    OrtCUDAProviderOptions cuda_options;
    std::memset(&cuda_options, 0, sizeof(cuda_options));
    Ort::ThrowOnError(
      OrtSessionOptionsAppendExecutionProvider_CUDA(session_options_, &cuda_options));
  }
#endif

  tools::logger()->info("[YOLO_0526_ORT] Loading ONNX model from {}", model_path_);
  session_ = std::make_unique<Ort::Session>(env_, model_path_.c_str(), session_options_);

  // Cache input / output names
  Ort::AllocatorWithDefaultOptions allocator;
  const size_t num_input_nodes = session_->GetInputCount();
  const size_t num_output_nodes = session_->GetOutputCount();

  input_names_.reserve(num_input_nodes);
  for (size_t i = 0; i < num_input_nodes; ++i) {
    auto name = session_->GetInputNameAllocated(i, allocator);
    input_names_.push_back(name.release());
  }

  output_names_.reserve(num_output_nodes);
  for (size_t i = 0; i < num_output_nodes; ++i) {
    auto name = session_->GetOutputNameAllocated(i, allocator);
    output_names_.push_back(name.release());
  }

  tools::logger()->info(
    "[YOLO_0526_ORT] Model loaded, inputs: {}, outputs: {}",
    num_input_nodes, num_output_nodes);
}

void YOLO_0526_ORT::letterbox(
  const cv::Mat & img, cv::Mat & out,
  float & scale, int & pad_w, int & pad_h) const
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

  out = cv::Mat(cv::Size(INPUT_W, INPUT_H), CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(out(cv::Rect(pad_w, pad_h, new_w, new_h)));
}

std::list<Armor> YOLO_0526_ORT::detect(const cv::Mat & raw_img, int /*frame_count*/)
{
  if (raw_img.empty()) {
    tools::logger()->warn("[YOLO_0526_ORT] Empty image!");
    return {};
  }

  // Preprocess: letterbox → RGB → float32/255 → NCHW
  float scale = 1.0f;
  int pad_w = 0;
  int pad_h = 0;
  cv::Mat padded;
  letterbox(raw_img, padded, scale, pad_w, pad_h);

  cv::Mat rgb;
  cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

  std::vector<float> input_blob(3 * INPUT_H * INPUT_W);
  const int c_step = INPUT_H * INPUT_W;

  for (int y = 0; y < INPUT_H; ++y) {
    const auto * row_ptr = rgb.ptr<cv::Vec3b>(y);
    for (int x = 0; x < INPUT_W; ++x) {
      const cv::Vec3b & p = row_ptr[x];
      // p = (R, G, B)
      input_blob[0 * c_step + y * INPUT_W + x] = static_cast<float>(p[0]) / 255.0f;
      input_blob[1 * c_step + y * INPUT_W + x] = static_cast<float>(p[1]) / 255.0f;
      input_blob[2 * c_step + y * INPUT_W + x] = static_cast<float>(p[2]) / 255.0f;
    }
  }

  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
    OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

  std::array<int64_t, 4> input_shape = {1, 3, INPUT_H, INPUT_W};
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    mem_info, input_blob.data(), input_blob.size(),
    input_shape.data(), input_shape.size());

  auto output_tensors = session_->Run(
    Ort::RunOptions{nullptr},
    input_names_.data(), &input_tensor, 1,
    output_names_.data(), output_names_.size());

  if (output_tensors.empty()) {
    tools::logger()->warn("[YOLO_0526_ORT] No output tensors!");
    return {};
  }

  Ort::Value & out_tensor = output_tensors[0];
  float * out_data = out_tensor.GetTensorMutableData<float>();
  auto type_info = out_tensor.GetTensorTypeAndShapeInfo();
  auto out_shape = type_info.GetShape();

  if (out_shape.size() != 3) {
    tools::logger()->warn(
      "[YOLO_0526_ORT] Unexpected output rank: {}", static_cast<int>(out_shape.size()));
    return {};
  }

  // Expected: (1, 25200, 23)
  const size_t num_rows = static_cast<size_t>(out_shape[1]);

  return decode(
    out_data, num_rows, conf_thresh_,
    scale, pad_w, pad_h, raw_img.size());
}

std::list<Armor> YOLO_0526_ORT::decode(
  const float * output, size_t num_rows,
  float conf_thresh,
  float scale, int pad_w, int pad_h,
  const cv::Size & orig_size) const
{
  // Per row layout:
  // [0:8]   4 keypoints (x0,y0,x1,y1,x2,y2,x3,y3)
  // [8]     confidence
  // [9:13]  color scores (4)
  // [13:23] digit scores (10)
  const int stride = 23;

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
    const float * row = output + i * stride;
    const float conf = row[8];
    if (conf < conf_thresh) {
      continue;
    }

    std::vector<cv::Point2f> kps;
    kps.reserve(4);

    float min_x = 1e9f;
    float min_y = 1e9f;
    float max_x = -1e9f;
    float max_y = -1e9f;

    for (int k = 0; k < 4; ++k) {
      float px = row[2 * k + 0];
      float py = row[2 * k + 1];

      // Undo letterbox (same as Python)
      float x = (px - static_cast<float>(pad_w)) / scale;
      float y = (py - static_cast<float>(pad_h)) / scale;

      x = std::clamp(x, 0.0f, static_cast<float>(img_w - 1));
      y = std::clamp(y, 0.0f, static_cast<float>(img_h - 1));

      min_x = std::min(min_x, x);
      min_y = std::min(min_y, y);
      max_x = std::max(max_x, x);
      max_y = std::max(max_y, y);

      kps.emplace_back(x, y);
    }

    if (max_x <= min_x || max_y <= min_y) {
      continue;
    }

    // Color argmax over [9:13)
    int best_color = 0;
    float best_color_score = row[9];
    for (int c = 1; c < 4; ++c) {
      float s = row[9 + c];
      if (s > best_color_score) {
        best_color_score = s;
        best_color = c;
      }
    }

    // Digit argmax over [13:23)
    int best_digit = 0;
    float best_digit_score = row[13];
    for (int d = 1; d < 10; ++d) {
      float s = row[13 + d];
      if (s > best_digit_score) {
        best_digit_score = s;
        best_digit = d;
      }
    }

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

  std::list<Armor> armors;
  armors.clear();

  for (size_t i = 0; i < boxes.size(); ++i) {
    armors.emplace_back(
      color_ids[i], digit_ids[i], confidences[i],
      boxes[i], all_keypoints[i]);
  }

  return armors;
}

}  // namespace auto_aim

