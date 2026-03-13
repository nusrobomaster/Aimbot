#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>

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

void capture_loop(
  const std::string & config_path, const std::string & output_folder)
{
  auto yaml = YAML::LoadFile(config_path);
  auto pattern_cols = yaml["pattern_cols"].as<int>();
  auto pattern_rows = yaml["pattern_rows"].as<int>();
  cv::Size pattern_size(pattern_cols, pattern_rows);

  io::Camera camera(config_path);
  H30IMU imu("/dev/ttyACM1", 460800);
  if (!imu.start()) {
    tools::logger()->error("Failed to start H30 IMU");
    return;
  }

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;

  // Let IMU stabilize
  tools::logger()->info("Waiting for IMU to stabilize...");
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  int count = 0;
  while (true) {
    camera.read(img, timestamp);
    Eigen::Quaterniond q = imu.getQuaternionAt(timestamp);

    // 在图像上显示欧拉角
    auto img_with_ypr = img.clone();
    Eigen::Vector3d zyx = tools::eulers(q, 2, 1, 0) * 57.3;  // degree
    tools::draw_text(img_with_ypr, fmt::format("Z {:.2f}", zyx[0]), {40, 40}, {0, 0, 255});
    tools::draw_text(img_with_ypr, fmt::format("Y {:.2f}", zyx[1]), {40, 80}, {0, 0, 255});
    tools::draw_text(img_with_ypr, fmt::format("X {:.2f}", zyx[2]), {40, 120}, {0, 0, 255});
    tools::draw_text(img_with_ypr, fmt::format("Saved: {}", count), {40, 160}, {0, 255, 0});

    // Detect chessboard corners (FAST_CHECK skips expensive search when board not visible)
    std::vector<cv::Point2f> corners;
    auto success = cv::findChessboardCorners(img, pattern_size, corners,
      cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
    if (success) {
      cv::drawChessboardCorners(img_with_ypr, pattern_size, corners, success);
    }
    cv::resize(img_with_ypr, img_with_ypr, {}, 0.5, 0.5);

    // 按"s"保存图片和对应四元数，按"q"退出程序
    cv::imshow("Press s to save, q to quit", img_with_ypr);
    auto key = cv::waitKey(1);
    if (key == 'q')
      break;
    else if (key != 's')
      continue;

    if (!success) {
      tools::logger()->warn("Chessboard not detected, not saving");
      continue;
    }

    // 保存图片和四元数
    count++;
    auto img_path = fmt::format("{}/{}.jpg", output_folder, count);
    auto q_path = fmt::format("{}/{}.txt", output_folder, count);
    cv::imwrite(img_path, img);
    write_q(q_path, q);
    tools::logger()->info("[{}] Saved in {}", count, output_folder);
  }
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

  // 新建输出文件夹
  std::filesystem::create_directory(output_folder);

  tools::logger()->info("Chessboard pattern: read from config");
  tools::logger()->info("Press 's' to save frame, 'q' to quit");
  // 主循环，保存图片和对应四元数
  capture_loop(config_path, output_folder);

  tools::logger()->warn("注意四元数输出顺序为wxyz");

  return 0;
}
