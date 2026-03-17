#pragma once

#include <list>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <NvInfer.h>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/yolo.hpp"

namespace auto_aim
{

class YOLO_0526_TRT : public YOLOBase
{
public:
  explicit YOLO_0526_TRT(const std::string& config_path, bool debug = false);
  ~YOLO_0526_TRT();

  std::list<Armor> detect(const cv::Mat& raw_img, int frame_count = 0) override;

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

  bool   debug_ = false;
  std::string model_path_;
  float  conf_thresh_ = 0.8f;

  // TensorRT
  nvinfer1::IRuntime*       runtime_   = nullptr;
  nvinfer1::ICudaEngine*    engine_    = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;

  int    input_index_  = -1;
  int    output_index_ = -1;
  size_t input_byte_   = 0;
  size_t output_byte_  = 0;

  void*  d_input_  = nullptr;
  void*  d_output_ = nullptr;
  std::vector<float> h_output_;

  // Pre-allocated host buffers (avoid per-frame allocation)
  std::vector<uint16_t>     input_blob_;   // FP16
  mutable cv::Mat           letterbox_buf_;
  mutable cv::Mat           rgb_buf_;
};

}  // namespace auto_aim
