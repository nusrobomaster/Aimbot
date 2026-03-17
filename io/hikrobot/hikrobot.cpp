#include "hikrobot.hpp"

#include <libusb-1.0/libusb.h>

#include "tools/logger.hpp"

using namespace std::chrono_literals;

namespace io
{
HikRobot::HikRobot(double exposure_ms, double gain, const std::string & vid_pid)
: exposure_us_(exposure_ms * 1e3), gain_(gain), queue_(1), daemon_quit_(false), vid_(-1), pid_(-1),
  use_bayer_mvs_convert_(true), acquisition_frame_rate_(165.0)
{
  set_vid_pid(vid_pid);
  if (libusb_init(NULL)) tools::logger()->warn("Unable to init libusb!");

  daemon_thread_ = std::thread{[this] {
    tools::logger()->info("HikRobot's daemon thread started.");

    capture_start();

    while (!daemon_quit_) {
      std::this_thread::sleep_for(100ms);

      if (capturing_) continue;

      capture_stop();
      reset_usb();
      capture_start();
    }

    capture_stop();

    tools::logger()->info("HikRobot's daemon thread stopped.");
  }};
}

HikRobot::HikRobot(
  double exposure_ms, double gain, const std::string & vid_pid, bool use_bayer_mvs_convert,
  double acquisition_frame_rate)
: exposure_us_(exposure_ms * 1e3), gain_(gain), queue_(1), daemon_quit_(false), vid_(-1), pid_(-1),
  use_bayer_mvs_convert_(use_bayer_mvs_convert), acquisition_frame_rate_(acquisition_frame_rate)
{
  set_vid_pid(vid_pid);
  if (libusb_init(NULL)) tools::logger()->warn("Unable to init libusb!");

  daemon_thread_ = std::thread{[this] {
    tools::logger()->info("HikRobot's daemon thread started.");

    capture_start();

    while (!daemon_quit_) {
      std::this_thread::sleep_for(100ms);

      if (capturing_) continue;

      capture_stop();
      reset_usb();
      capture_start();
    }

    capture_stop();

    tools::logger()->info("HikRobot's daemon thread stopped.");
  }};
}

HikRobot::~HikRobot()
{
  daemon_quit_ = true;
  if (daemon_thread_.joinable()) daemon_thread_.join();
  tools::logger()->info("HikRobot destructed.");
}

void HikRobot::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  CameraData data;
  queue_.pop(data);

  img = data.img;
  timestamp = data.timestamp;
}

void HikRobot::capture_start()
{
  capturing_ = false;
  capture_quit_ = false;

  unsigned int ret;

  MV_CC_DEVICE_INFO_LIST device_list;
  ret = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_EnumDevices failed: {:#x}", ret);
    return;
  }

  if (device_list.nDeviceNum == 0) {
    tools::logger()->warn("Not found camera!");
    return;
  }

  ret = MV_CC_CreateHandle(&handle_, device_list.pDeviceInfo[0]);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_CreateHandle failed: {:#x}", ret);
    return;
  }

  ret = MV_CC_OpenDevice(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_OpenDevice failed: {:#x}", ret);
    return;
  }

  set_enum_value("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS);
  set_enum_value("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
  set_enum_value("GainAuto", MV_GAIN_MODE_OFF);
  set_float_value("ExposureTime", exposure_us_);
  set_float_value("Gain", gain_);

  // pb2025-style: use AcquisitionFrameRateEnable + AcquisitionFrameRate (165 Hz default)
  ret = MV_CC_SetBoolValue(handle_, "AcquisitionFrameRateEnable", true);
  if (ret == MV_OK) {
    ret = MV_CC_SetFloatValue(handle_, "AcquisitionFrameRate", acquisition_frame_rate_);
    if (ret == MV_OK) {
      tools::logger()->info("Acquisition frame rate set to {:.1f} Hz", acquisition_frame_rate_);
    }
  }
  if (ret != MV_OK) {
    MV_CC_SetFrameRate(handle_, static_cast<float>(acquisition_frame_rate_));
  }

  // pb2025-style: set BayerRG8 for faster transfer + MVS SDK conversion
  if (use_bayer_mvs_convert_) {
    ret = MV_CC_SetEnumValueByString(handle_, "ADCBitDepth", "Bits_8");
    if (ret != MV_OK) {
      tools::logger()->warn("MV_CC_SetEnumValueByString(ADCBitDepth) failed: {:#x}", ret);
    }
    ret = MV_CC_SetEnumValueByString(handle_, "PixelFormat", "BayerRG8");
    if (ret == MV_OK) {
      tools::logger()->info("Pixel format set to BayerRG8 (using MVS ConvertPixelType)");
    } else {
      tools::logger()->warn("BayerRG8 not supported, falling back to cv::cvtColor");
      use_bayer_mvs_convert_ = false;
    }
  }

  ret = MV_CC_GetImageInfo(handle_, &img_info_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_GetImageInfo failed: {:#x}", ret);
  } else if (use_bayer_mvs_convert_) {
    rgb_buffer_.resize(img_info_.nHeightMax * img_info_.nWidthMax * 3);
    convert_param_.nWidth = img_info_.nWidthValue;
    convert_param_.nHeight = img_info_.nHeightValue;
    convert_param_.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
  }

  ret = MV_CC_StartGrabbing(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_StartGrabbing failed: {:#x}", ret);
    return;
  }

  capture_thread_ = std::thread{[this] {
    tools::logger()->info("HikRobot's capture thread started.");

    capturing_ = true;

    MV_FRAME_OUT raw;
    const unsigned int get_buffer_timeout_ms = 1000;

    while (!capture_quit_) {
      // pb2025-style: NO sleep - run as fast as camera delivers frames
      unsigned int ret = MV_CC_GetImageBuffer(handle_, &raw, get_buffer_timeout_ms);
      if (ret != MV_OK) {
        tools::logger()->warn("MV_CC_GetImageBuffer failed: {:#x}", ret);
        break;
      }

      auto timestamp = std::chrono::steady_clock::now();
      const auto & frame_info = raw.stFrameInfo;
      auto pixel_type = frame_info.enPixelType;

      cv::Mat dst_image;

      if (use_bayer_mvs_convert_ && rgb_buffer_.size() >= frame_info.nWidth * frame_info.nHeight * 3) {
        // pb2025-style: use MVS SDK ConvertPixelType (faster than cv::cvtColor)
        convert_param_.enSrcPixelType = pixel_type;
        convert_param_.pSrcData = raw.pBufAddr;
        convert_param_.nSrcDataLen = frame_info.nFrameLen;
        convert_param_.pDstBuffer = rgb_buffer_.data();
        convert_param_.nDstBufferSize = static_cast<unsigned int>(rgb_buffer_.size());
        convert_param_.nWidth = frame_info.nWidth;
        convert_param_.nHeight = frame_info.nHeight;

        ret = MV_CC_ConvertPixelType(handle_, &convert_param_);
        if (ret == MV_OK) {
          dst_image = cv::Mat(
            frame_info.nHeight, frame_info.nWidth, CV_8UC3, rgb_buffer_.data()).clone();
        } else {
          use_bayer_mvs_convert_ = false;
          tools::logger()->warn("MV_CC_ConvertPixelType failed: {:#x}, falling back to cv::cvtColor", ret);
        }
      }

      if (!use_bayer_mvs_convert_ || dst_image.empty()) {
        // Fallback: OpenCV conversion (original Aimbot path)
        int cv_type;
        if (pixel_type == PixelType_Gvsp_BGR8_Packed ||
            pixel_type == PixelType_Gvsp_RGB8_Packed) {
          cv_type = CV_8UC3;
        } else if (pixel_type == PixelType_Gvsp_YUV422_Packed) {
          cv_type = CV_8UC2;
        } else {
          cv_type = CV_8UC1;
        }
        cv::Mat img(cv::Size(frame_info.nWidth, frame_info.nHeight), cv_type, raw.pBufAddr);

        static bool first_frame = true;
        if (first_frame) {
          tools::logger()->info(
            "Camera pixel type: {:#x}, Mat type: CV_8UC{}",
            static_cast<unsigned int>(pixel_type), CV_MAT_CN(cv_type));
          first_frame = false;
        }

        const static std::unordered_map<MvGvspPixelType, cv::ColorConversionCodes> type_map = {
          {PixelType_Gvsp_BayerGR8, cv::COLOR_BayerGR2BGR},
          {PixelType_Gvsp_BayerRG8, cv::COLOR_BayerRG2BGR},
          {PixelType_Gvsp_BayerGB8, cv::COLOR_BayerGB2BGR},
          {PixelType_Gvsp_BayerBG8, cv::COLOR_BayerBG2BGR}};
        auto it = type_map.find(pixel_type);
        if (it == type_map.end()) {
          if (pixel_type == PixelType_Gvsp_BGR8_Packed) {
            dst_image = img.clone();
          } else if (pixel_type == PixelType_Gvsp_RGB8_Packed) {
            cv::cvtColor(img, dst_image, cv::COLOR_RGB2BGR);
          } else if (pixel_type == PixelType_Gvsp_Mono8) {
            cv::cvtColor(img, dst_image, cv::COLOR_GRAY2BGR);
          } else if (pixel_type == PixelType_Gvsp_YUV422_Packed) {
            cv::cvtColor(img, dst_image, cv::COLOR_YUV2BGR_YUYV);
          } else {
            tools::logger()->error(
              "Unsupported pixel type: {:#x}", static_cast<unsigned int>(pixel_type));
            MV_CC_FreeImageBuffer(handle_, &raw);
            continue;
          }
        } else {
          cv::cvtColor(img, dst_image, it->second);
        }
      }

      queue_.push({dst_image, timestamp});

      ret = MV_CC_FreeImageBuffer(handle_, &raw);
      if (ret != MV_OK) {
        tools::logger()->warn("MV_CC_FreeImageBuffer failed: {:#x}", ret);
        break;
      }
    }

    capturing_ = false;
    tools::logger()->info("HikRobot's capture thread stopped.");
  }};
}

void HikRobot::capture_stop()
{
  capture_quit_ = true;
  if (capture_thread_.joinable()) capture_thread_.join();

  unsigned int ret;

  ret = MV_CC_StopGrabbing(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_StopGrabbing failed: {:#x}", ret);
    return;
  }

  ret = MV_CC_CloseDevice(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_CloseDevice failed: {:#x}", ret);
    return;
  }

  ret = MV_CC_DestroyHandle(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_DestroyHandle failed: {:#x}", ret);
    return;
  }
}

void HikRobot::set_float_value(const std::string & name, double value)
{
  unsigned int ret;

  ret = MV_CC_SetFloatValue(handle_, name.c_str(), value);

  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_SetFloatValue(\"{}\", {}) failed: {:#x}", name, value, ret);
    return;
  }
}

void HikRobot::set_enum_value(const std::string & name, unsigned int value)
{
  unsigned int ret;

  ret = MV_CC_SetEnumValue(handle_, name.c_str(), value);

  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_SetEnumValue(\"{}\", {}) failed: {:#x}", name, value, ret);
    return;
  }
}

void HikRobot::set_vid_pid(const std::string & vid_pid)
{
  auto index = vid_pid.find(':');
  if (index == std::string::npos) {
    tools::logger()->warn("Invalid vid_pid: \"{}\"", vid_pid);
    return;
  }

  auto vid_str = vid_pid.substr(0, index);
  auto pid_str = vid_pid.substr(index + 1);

  try {
    vid_ = std::stoi(vid_str, 0, 16);
    pid_ = std::stoi(pid_str, 0, 16);
  } catch (const std::exception &) {
    tools::logger()->warn("Invalid vid_pid: \"{}\"", vid_pid);
  }
}

void HikRobot::reset_usb() const
{
  if (vid_ == -1 || pid_ == -1) return;

  // https://github.com/ralight/usb-reset/blob/master/usb-reset.c
  auto handle = libusb_open_device_with_vid_pid(NULL, vid_, pid_);
  if (!handle) {
    tools::logger()->warn("Unable to open usb!");
    return;
  }

  if (libusb_reset_device(handle))
    tools::logger()->warn("Unable to reset usb!");
  else
    tools::logger()->info("Reset usb successfully :)");

  libusb_close(handle);
}

}  // namespace io
