#include "camera.hpp"

#include <stdexcept>

#include "hikrobot/hikrobot.hpp"
#include "mindvision/mindvision.hpp"
#include "tools/yaml.hpp"

namespace io
{
Camera::Camera(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto camera_name = tools::read<std::string>(yaml, "camera_name");
  auto exposure_ms = tools::read<double>(yaml, "exposure_ms");

  if (camera_name == "mindvision") {
    auto gamma = tools::read<double>(yaml, "gamma");
    auto vid_pid = tools::read<std::string>(yaml, "vid_pid");
    camera_ = std::make_unique<MindVision>(exposure_ms, gamma, vid_pid);
  }

  else if (camera_name == "hikrobot") {
    auto gain = tools::read<double>(yaml, "gain");
    auto vid_pid = tools::read<std::string>(yaml, "vid_pid");
    bool use_bayer_mvs_convert = true;
    double acquisition_frame_rate = 165.0;
    if (yaml["use_bayer_mvs_convert"]) {
      use_bayer_mvs_convert = yaml["use_bayer_mvs_convert"].as<bool>();
    }
    if (yaml["acquisition_frame_rate"]) {
      acquisition_frame_rate = yaml["acquisition_frame_rate"].as<double>();
    }
    camera_ = std::make_unique<HikRobot>(
      exposure_ms, gain, vid_pid, use_bayer_mvs_convert, acquisition_frame_rate);
  }

  else {
    throw std::runtime_error("Unknow camera_name: " + camera_name + "!");
  }
}

void Camera::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  camera_->read(img, timestamp);
}

}  // namespace io