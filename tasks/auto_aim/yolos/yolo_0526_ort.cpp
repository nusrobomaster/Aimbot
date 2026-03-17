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
    OrtTensorRTProviderOptionsV2* trt_options;
    Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&trt_options));
    
    std::vector<const char*> keys{
      "device_id",
      "trt_fp16_enable",
      "trt_engine_cache_enable",
      "trt_engine_cache_path"
    };
    std::vector<const char*> values{
      "0",
      "1",
      "1",
      "/home/slavito2/Code/Aimbot/assets/trt_cache"
    };
    Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptions(
      trt_options, keys.data(), values.data(), keys.size()));
    session_options_.AppendExecutionProvider_TensorRT_V2(*trt_options);
    Ort::GetApi().ReleaseTensorRTProviderOptions(trt_options);
  }
  #endif
    // Try to enable CUDA EP if available
  #if defined(ORT_CUDA_EXECUTION_PROVIDER)
    {
      Ort::ThrowOnError(
        OrtSessionOptionsAppendExecutionProvider_CUDA(session_options_, 0));
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

  // Pre-allocate buffers to avoid per-frame allocation in detect()
  input_blob_.resize(3 * INPUT_H * INPUT_W);
  letterbox_buf_.create(cv::Size(INPUT_W, INPUT_H), CV_8UC3);
  rgb_buf_.create(cv::Size(INPUT_W, INPUT_H), CV_8UC3);
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

  if (out.cols != INPUT_W || out.rows != INPUT_H || out.type() != CV_8UC3)
    out.create(cv::Size(INPUT_W, INPUT_H), CV_8UC3);
  out.setTo(cv::Scalar(114, 114, 114));
  resized.copyTo(out(cv::Rect(pad_w, pad_h, new_w, new_h)));
}

std::list<Armor> YOLO_0526_ORT::detect(const cv::Mat & raw_img, int /*frame_count*/)
{
  if (raw_img.empty()) {
    tools::logger()->warn("[YOLO_0526_ORT] Empty image!");
    return {};
  }

  // Preprocess: letterbox → RGB → float16/255 → NCHW (use pre-allocated buffers)
  float scale = 1.0f;
  int pad_w = 0;
  int pad_h = 0;
  letterbox(raw_img, letterbox_buf_, scale, pad_w, pad_h);
  cv::cvtColor(letterbox_buf_, rgb_buf_, cv::COLOR_BGR2RGB);

  const int c_step = INPUT_H * INPUT_W;
  for (int y = 0; y < INPUT_H; ++y) {
    const auto * row_ptr = rgb_buf_.ptr<cv::Vec3b>(y);
    for (int x = 0; x < INPUT_W; ++x) {
      const cv::Vec3b & p = row_ptr[x];
      input_blob_[0 * c_step + y * INPUT_W + x] = Ort::Float16_t(static_cast<float>(p[0]) / 255.0f);
      input_blob_[1 * c_step + y * INPUT_W + x] = Ort::Float16_t(static_cast<float>(p[1]) / 255.0f);
      input_blob_[2 * c_step + y * INPUT_W + x] = Ort::Float16_t(static_cast<float>(p[2]) / 255.0f);
    }
  }

  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
    OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

  std::array<int64_t, 4> input_shape = {1, 3, INPUT_H, INPUT_W};
  Ort::Value input_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
    mem_info, input_blob_.data(), input_blob_.size(),
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
  auto type_info = out_tensor.GetTensorTypeAndShapeInfo();
  auto out_shape = type_info.GetShape();

  if (out_shape.size() != 3) {
    tools::logger()->warn(
      "[YOLO_0526_ORT] Unexpected output rank: {}", static_cast<int>(out_shape.size()));
    return {};
  }

  float * out_data = out_tensor.GetTensorMutableData<float>();

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
    const float * row = output + i * stride;
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
      if (s > best_color_score) {
        best_color_score = s;
        best_color = c;
      }
    }

    int best_digit = 0;
    float best_digit_score = row[13];
    for (int d = 1; d < 9; ++d) {
      float s = row[13 + d];
      if (s > best_digit_score) {
        best_digit_score = s;
        best_digit = d;
      }
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