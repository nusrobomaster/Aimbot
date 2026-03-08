#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/h30_imu/h30_imu.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

const std::string keys =
  "{help h usage ?  |                          | 输出命令行参数说明}"
  "{@config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
  "{output-folder o |      assets/img_with_q   | 输出文件夹路径   }";

void write_q(const std::string q_path, const Eigen::Quaterniond & q)
{
  std::ofstream q_file(q_path);
  Eigen::Vector4d xyzw = q.coeffs();
  // 输出顺序为wxyz
  q_file << fmt::format("{} {} {} {}", xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
  q_file.close();
}

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);
  auto output_folder = cli.get<std::string>("output-folder");

  // 读取标定板参数
  auto yaml = YAML::LoadFile(config_path);
  int pattern_cols = yaml["pattern_cols"].as<int>();
  int pattern_rows = yaml["pattern_rows"].as<int>();
  cv::Size pattern_size(pattern_cols, pattern_rows);

  // 新建输出文件夹
  std::filesystem::create_directories(output_folder);

  // 初始化相机和IMU
  io::Camera camera(config_path);
  // Start camera first to verify it works
  cv::Mat test_img;
  std::chrono::steady_clock::time_point test_ts;
  camera.read(test_img, test_ts);
  if (test_img.empty()) {
    tools::logger()->error("Camera failed to read - check vid_pid and connection");
    return 1;
  }
  fmt::print("[Camera] OK - {}x{}\n", test_img.cols, test_img.rows);

  std::string imu_port = yaml["imu_port"] ? yaml["imu_port"].as<std::string>() : "/dev/ttyACM1";
  int imu_baud = yaml["imu_baud"] ? yaml["imu_baud"].as<int>() : 460800;
  fmt::print("[IMU] Opening {} at {} baud\n", imu_port, imu_baud);

  H30IMU imu(imu_port, imu_baud);
  if (!imu.start()) {
    tools::logger()->error("Failed to start H30 IMU on {}", imu_port);
    return 1;
  }

  // IMU tare: 采集零参考四元数 (保持云台静止水平)
  fmt::print("[IMU] Capturing zero reference (keep gimbal STILL and LEVEL)...\n");
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  Eigen::Quaterniond q_zero = Eigen::Quaterniond::Identity();
  {
    const int TARE_SAMPLES = 100;
    Eigen::Vector4d q_sum = Eigen::Vector4d::Zero();
    for (int i = 0; i < TARE_SAMPLES; ++i) {
      auto tp = std::chrono::steady_clock::now();
      Eigen::Quaterniond qs = imu.getQuaternionAt(tp);
      Eigen::Vector4d c = qs.coeffs();
      if (i == 0) {
        q_sum = c;
      } else {
        if (c.dot(q_sum) < 0) c = -c;
        q_sum += c;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    q_zero.coeffs() = q_sum.normalized();
  }
  fmt::print("[IMU] Zero reference captured: q0 = ({}, {}, {}, {})\n",
             q_zero.w(), q_zero.x(), q_zero.y(), q_zero.z());

  fmt::print("\nBoard: {}x{} inner corners (chessboard)\n", pattern_cols, pattern_rows);
  fmt::print("Output: {}\n", output_folder);
  fmt::print("Press 's' to save, 'q' to quit\n\n");

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  int count = 0;

  while (true) {
    camera.read(img, timestamp);
    Eigen::Quaterniond q_raw = imu.getQuaternionAt(timestamp);

    // 显示用的欧拉角 (tared, 仅用于验证)
    Eigen::Quaterniond q_tared = q_zero.conjugate() * q_raw;
    Eigen::Vector3d zyx = tools::eulers(q_tared, 2, 1, 0) * 57.3;

    auto drawing = img.clone();
    tools::draw_text(drawing, fmt::format("yaw   {:.2f}", zyx[0]), {40, 40}, {0, 0, 255});
    tools::draw_text(drawing, fmt::format("pitch {:.2f}", zyx[1]), {40, 80}, {0, 0, 255});
    tools::draw_text(drawing, fmt::format("roll  {:.2f}", zyx[2]), {40, 120}, {0, 0, 255});
    tools::draw_text(drawing, fmt::format("saved: {}", count), {40, 160}, {0, 255, 0});

    // 检测棋盘格 (downscale for speed, refine on full-res)
    constexpr double SCALE = 0.5;
    cv::Mat small;
    cv::resize(img, small, {}, SCALE, SCALE);

    std::vector<cv::Point2f> corners_2d;
    auto success = cv::findChessboardCorners(small, pattern_size, corners_2d,
      cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
    if (success) {
      // Scale corners back to full resolution
      for (auto & c : corners_2d) { c.x /= SCALE; c.y /= SCALE; }
      cv::Mat gray;
      cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
      cv::cornerSubPix(gray, corners_2d, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
    }
    cv::drawChessboardCorners(drawing, pattern_size, corners_2d, success);

    cv::resize(drawing, drawing, {}, 0.5, 0.5);
    cv::imshow("Press s to save, q to quit", drawing);
    auto key = cv::waitKey(1);
    if (key == 'q')
      break;
    else if (key != 's')
      continue;

    if (!success) {
      tools::logger()->warn("Chessboard not detected, not saving");
      continue;
    }

    // 保存图片和原始四元数 (未tare, calibrate_robotworld_handeye自行处理)
    count++;
    auto img_path = fmt::format("{}/{}.jpg", output_folder, count);
    auto q_path = fmt::format("{}/{}.txt", output_folder, count);
    cv::imwrite(img_path, img);
    write_q(q_path, q_raw);
    tools::logger()->info("[{}] Saved in {}", count, output_folder);
  }

  tools::logger()->warn("注意四元数输出顺序为wxyz (raw, 未tare)");

  return 0;
}
