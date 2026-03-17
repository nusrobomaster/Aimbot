#pragma once

#include <list>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/yolo.hpp"   // YOLOBase

namespace auto_aim
{

class YOLO_0526_ORT : public YOLOBase
{
public:
  explicit YOLO_0526_ORT(const std::string& config_path, bool debug = false);

  // YOLOBase interface
  std::list<Armor> detect(const cv::Mat& raw_img, int frame_count = 0) override;

  // Not used by this backend
  std::list<Armor> postprocess(
    double /*scale*/, cv::Mat& /*output*/,
    const cv::Mat& /*bgr_img*/, int /*frame_count*/) override { return {}; }

private:
  static constexpr int INPUT_W = 640;
  static constexpr int INPUT_H = 640;

  void letterbox(const cv::Mat& img, cv::Mat& out,
                 float& scale, int& pad_w, int& pad_h) const;

  std::list<Armor> decode(const float* output, size_t num_rows,
                          float conf_thresh, float scale,
                          int pad_w, int pad_h,
                          const cv::Size& orig_size) const;

  bool        debug_;
  std::string model_path_;
  float       conf_thresh_;

  // ORT objects
  Ort::Env                        env_;
  Ort::SessionOptions             session_options_;
  std::unique_ptr<Ort::Session>   session_;
  std::vector<const char*>        input_names_;
  std::vector<const char*>        output_names_;

  // Pre-allocated buffers — avoids per-frame heap allocation
  mutable std::vector<Ort::Float16_t> input_blob_;
  mutable cv::Mat                     letterbox_buf_;
  mutable cv::Mat                     rgb_buf_;
};

}  // namespace auto_aim