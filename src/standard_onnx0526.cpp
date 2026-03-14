#include <fmt/core.h>
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <iomanip>
#include <iostream>
#include <unistd.h>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <std_msgs/msg/bool.hpp>

#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "io/h30_imu/h30_imu.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

using namespace std::chrono;

// Default config for ONNX 0526 (TensorRT/CUDA) pipeline
const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/standard_onnx0526.yaml | 位置参数，yaml配置文件路径 }";

/**
 * @brief Draw armor plate with its keypoints and information
 */
void drawArmorPlate(cv::Mat& img, const auto_aim::Armor& armor, const cv::Scalar& color,
                   const std::string& label_prefix = "") {
    if (armor.points.size() >= 4) {
        for (int i = 0; i < 4; i++) {
            cv::line(img, armor.points[i], armor.points[(i + 1) % 4], color, 2);
        }
        for (const auto& point : armor.points) {
            cv::circle(img, point, 3, color, -1);
        }
        cv::Point2f center(0, 0);
        for (const auto& point : armor.points) {
            center.x += point.x;
            center.y += point.y;
        }
        center.x /= 4;
        center.y /= 4;
        cv::circle(img, center, 4, cv::Scalar(255, 255, 255), -1);
        std::string label;
        if (!label_prefix.empty()) label += label_prefix + " ";
        label += auto_aim::ARMOR_NAMES[armor.name] + " ";
        label += "(" + auto_aim::ARMOR_TYPES[armor.type] + ") ";
        label += fmt::format("{:.2f}", armor.confidence);
        cv::putText(img, label, cv::Point(center.x + 10, center.y),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        if (armor.xyz_in_world.norm() > 0) {
            std::string pos_text = fmt::format("({:.2f},{:.2f},{:.2f})m",
                armor.xyz_in_world[0], armor.xyz_in_world[1], armor.xyz_in_world[2]);
            cv::putText(img, pos_text, cv::Point(center.x + 10, center.y + 20),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
        }
        cv::line(img, armor.left.top, armor.left.bottom, cv::Scalar(0, 255, 255), 2);
        cv::line(img, armor.right.top, armor.right.bottom, cv::Scalar(0, 255, 255), 2);
    }
    else if (armor.box.area() > 0) {
        cv::rectangle(img, armor.box, color, 2);
        cv::Point center(armor.box.x + armor.box.width/2, armor.box.y + armor.box.height/2);
        cv::circle(img, center, 3, color, -1);
        std::string label = label_prefix + " " + auto_aim::ARMOR_NAMES[armor.name];
        cv::putText(img, label, cv::Point(armor.box.x, armor.box.y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
}

void drawTarget(cv::Mat& img, const auto_aim::Target& target,
                auto_aim::Solver& solver, const cv::Scalar& color) {
    auto armor_xyza_list = target.armor_xyza_list();
    int armor_count = 0;
    for (const Eigen::Vector4d& xyza : armor_xyza_list) {
        auto image_points = solver.reproject_armor(
            xyza.head(3), xyza[3], target.armor_type, target.name);
        cv::Scalar armor_color = color;
        if (armor_count == 0) armor_color = cv::Scalar(0, 255, 0);
        else if (armor_count == 1) armor_color = cv::Scalar(255, 255, 0);
        for (int i = 0; i < 4; i++) {
            cv::line(img, image_points[i], image_points[(i + 1) % 4], armor_color, 2);
        }
        for (const auto& point : image_points) cv::circle(img, point, 3, armor_color, -1);
        cv::Point2f center(0, 0);
        for (const auto& point : image_points) { center.x += point.x; center.y += point.y; }
        center.x /= 4; center.y /= 4;
        cv::circle(img, center, 4, armor_color, -1);
        armor_count++;
    }
    if (!armor_xyza_list.empty()) {
        auto first_armor_points = solver.reproject_armor(
            armor_xyza_list[0].head(3), armor_xyza_list[0][3], target.armor_type, target.name);
        cv::Point2f text_pos(0, 0);
        for (const auto& point : first_armor_points) { text_pos.x += point.x; text_pos.y += point.y; }
        text_pos.x /= 4; text_pos.y /= 4;
        std::string target_info = fmt::format("{} (ID:{})", auto_aim::ARMOR_NAMES[target.name], target.last_id);
        cv::putText(img, target_info, cv::Point(text_pos.x + 15, text_pos.y),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        Eigen::VectorXd x = target.ekf_x();
        if (x.size() >= 11) {
            double distance = std::sqrt(x[0]*x[0] + x[2]*x[2] + x[4]*x[4]);
            std::string state_text = fmt::format("Dist: {:.1f}m v:{:.1f}m/s", distance, Eigen::Vector3d(x[1], x[3], x[5]).norm());
            cv::putText(img, state_text, cv::Point(text_pos.x + 15, text_pos.y + 20),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
        }
    }
}

void displayAutoAimResults(cv::Mat& img,
                          const std::list<auto_aim::Armor>& armors,
                          const std::list<auto_aim::Target>& targets,
                          auto_aim::Solver& solver,
                          const io::Command& command,
                          const Eigen::Vector3d& ypr,
                          const auto_aim::AimPoint& aim_point,
                          float display_scale = 0.75f) {
    cv::Mat display_img = img.clone();
    cv::putText(display_img, fmt::format("Armors: {} | Targets: {}", armors.size(), targets.size()),
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    cv::putText(display_img, fmt::format("Gimbal Yaw: {:.1f}° | Pitch: {:.1f}°", ypr[0]*180.0/M_PI, ypr[1]*180.0/M_PI),
                cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 100), 1);
    int armor_idx = 0;
    for (const auto& armor : armors) {
        cv::Scalar color;
        switch (armor.color) {
            case auto_aim::Color::red: color = cv::Scalar(0, 0, 255); break;
            case auto_aim::Color::blue: color = cv::Scalar(255, 0, 0); break;
            case auto_aim::Color::purple: color = cv::Scalar(255, 0, 255); break;
            case auto_aim::Color::extinguish: color = cv::Scalar(0, 255, 255); break;
            default: color = cv::Scalar(200, 200, 200);
        }
        if (armor.type == auto_aim::ArmorType::small)
            color = cv::Scalar(color[0]*1.2, color[1]*1.2, color[2]*1.2);
        drawArmorPlate(display_img, armor, color, fmt::format("D{}", armor_idx++));
    }
    for (const auto& target : targets) {
        cv::Scalar color;
        switch (target.name) {
            case auto_aim::ArmorName::outpost:
            case auto_aim::ArmorName::sentry: color = cv::Scalar(0, 165, 255); break;
            case auto_aim::ArmorName::base: color = cv::Scalar(255, 0, 255); break;
            default: color = cv::Scalar(0, 255, 255);
        }
        drawTarget(display_img, target, solver, color);
    }
    if (aim_point.valid) {
        auto image_points = solver.reproject_armor(aim_point.xyza.head(3), aim_point.xyza[3], auto_aim::ArmorType::small, auto_aim::ArmorName::one);
        for (int i = 0; i < 4; i++)
            cv::line(display_img, image_points[i], image_points[(i+1)%4], cv::Scalar(0,0,255), 2);
        cv::Point2f center(0,0);
        for (const auto& point : image_points) { center.x += point.x; center.y += point.y; }
        center.x /= 4; center.y /= 4;
        cv::circle(display_img, center, 8, cv::Scalar(0,0,255), -1);
        cv::circle(display_img, center, 12, cv::Scalar(0,0,255), 2);
        cv::putText(display_img, "AIM", cv::Point(center.x+15, center.y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);
    }
    cv::Point center(display_img.cols/2, display_img.rows/2);
    cv::line(display_img, cv::Point(center.x-20, center.y), cv::Point(center.x+20, center.y), cv::Scalar(0,255,0), 2);
    cv::line(display_img, cv::Point(center.x, center.y-20), cv::Point(center.x, center.y+20), cv::Scalar(0,255,0), 2);
    cv::circle(display_img, center, 8, cv::Scalar(0,255,0), 1);
    cv::putText(display_img, fmt::format("Cmd: Y{:.2f}° P{:.2f}°", command.yaw*180.0/M_PI, command.pitch*180.0/M_PI),
                cv::Point(10, display_img.rows-60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,255,0), 2);
    if (command.control) cv::putText(display_img, "CONTROL ACTIVE", cv::Point(10, display_img.rows-90), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,165,0), 2);
    if (command.shoot) cv::putText(display_img, "FIRE!", cv::Point(10, display_img.rows-30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,0,255), 3);
    if (command.control && (std::abs(command.yaw)>0.001 || std::abs(command.pitch)>0.001)) {
        int aim_x = center.x - static_cast<int>(command.yaw*180.0/M_PI*10);
        int aim_y = center.y + static_cast<int>(command.pitch*180.0/M_PI*10);
        cv::arrowedLine(display_img, center, cv::Point(aim_x, aim_y), cv::Scalar(255,0,255), 2, 8, 0, 0.1);
    }
    if (std::abs(display_scale - 1.0f) > 0.01f) cv::resize(display_img, display_img, cv::Size(), display_scale, display_scale);
    cv::imshow("Auto Aim Results", display_img);
    char key = cv::waitKey(1);
    if (key == 27) throw std::runtime_error("User pressed ESC to exit");
    else if (key == 's' || key == 'S') { static int save_count = 0; cv::imwrite(fmt::format("frame_{:04d}.png", save_count++), display_img); tools::logger()->info("Saved frame"); }
    else if (key == 'p' || key == 'P') { tools::logger()->info("Paused."); cv::waitKey(0); }
    else if (key == 'd' || key == 'D') {
        tools::logger()->info("Armors: {} Targets: {}", armors.size(), targets.size());
        tools::logger()->info("Cmd yaw={:.3f} pitch={:.3f} shoot={}", command.yaw*180.0/M_PI, command.pitch*180.0/M_PI, command.shoot);
    }
    else if (key == 'c' || key == 'C') system("clear");
}

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  auto ros_node = rclcpp::Node::make_shared("auto_aim_vision");
  auto aimbot_pub = ros_node->create_publisher<geometry_msgs::msg::Vector3>("aimbot_cmd", 10);
  auto fire_pub   = ros_node->create_publisher<std_msgs::msg::Bool>("fire_cmd", 10);
  std::thread ros_spin_thread([&ros_node]() { rclcpp::spin(ros_node); });
  tools::logger()->info("[ROS2] Node 'auto_aim_vision' started (ONNX 0526), publishing to aimbot_cmd / fire_cmd");

  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    rclcpp::shutdown();
    ros_spin_thread.join();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;
  io::Camera camera(config_path);
  H30IMU imu("/dev/ttyACM0", 460800);
  if (!imu.start()) tools::logger()->warn("Failed to start external IMU");
  auto_aim::YOLO detector(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  cv::namedWindow("Auto Aim Results", cv::WINDOW_NORMAL);
  cv::resizeWindow("Auto Aim Results", 1280, 720);
  auto last_time = std::chrono::steady_clock::now();
  int frame_count = 0;
  float fps = 0.0f;
  bool prev_fire_state = false;
  int send_count = 0, no_target_count = 0;

  std::cout << "\n=== Auto-Aim (ONNX 0526 / TensorRT) ===" << std::endl;
  std::cout << "Config: " << config_path << std::endl;
  std::cout << "ESC=exit, s=save, p=pause, d=debug, c=clear\n" << std::endl;

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  Eigen::Quaterniond q_zero = Eigen::Quaterniond::Identity();
  {
    Eigen::Vector4d q_sum = Eigen::Vector4d::Zero();
    for (int i = 0; i < 100; ++i) {
      auto tp = std::chrono::steady_clock::now();
      Eigen::Quaterniond qs = imu.getQuaternionAt(tp);
      Eigen::Vector4d c = qs.coeffs();
      if (i > 0 && c.dot(q_sum) < 0) c = -c;
      q_sum += c;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    q_zero.coeffs() = q_sum.normalized();
  }

  while (!exiter.exit()) {
    camera.read(img, t);
    Eigen::Quaterniond q_raw = imu.getQuaternionAt(t - 1ms);
    Eigen::Quaterniond q_tared = q_zero.conjugate() * q_raw;
    solver.set_R_gimbal2world(q_tared);
    Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    auto armors = detector.detect(img);
    auto targets = tracker.track(armors, t, true);
    auto command = aimer.aim(targets, t, 23.0);

    bool should_shoot = false;
    if (!targets.empty()) {
      Eigen::Vector3d gimbal_pos(command.yaw, command.pitch, 0);
      should_shoot = shooter.shoot(command, aimer, targets, gimbal_pos);
    }

    geometry_msgs::msg::Vector3 aim_msg;
    if (aimer.debug_aim_point.valid && !targets.empty()) {
      aim_msg.x = -tools::limit_rad(command.yaw - ypr[0]);
      aim_msg.y = -(command.pitch + ypr[1]);
      aim_msg.z = 0.0;
      send_count++;
      no_target_count = 0;
    } else {
      aim_msg.x = aim_msg.y = aim_msg.z = 0.0;
      no_target_count++;
    }
    aimbot_pub->publish(aim_msg);
    if (should_shoot != prev_fire_state) {
      std_msgs::msg::Bool fire_msg;
      fire_msg.data = should_shoot;
      fire_pub->publish(fire_msg);
      prev_fire_state = should_shoot;
    }

    try {
      displayAutoAimResults(img, armors, targets, solver, command, ypr, aimer.debug_aim_point);
    } catch (const std::exception& e) {
      tools::logger()->error("Display error: {}", e.what());
      break;
    }
    frame_count++;
    if (frame_count % 30 == 0) {
      auto now = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
      fps = 30000.0f / elapsed;
      tools::logger()->info("FPS: {:.1f}", fps);
      last_time = now;
    }
  }

  geometry_msgs::msg::Vector3 zero_aim;
  zero_aim.x = zero_aim.y = zero_aim.z = 0.0;
  aimbot_pub->publish(zero_aim);
  std_msgs::msg::Bool stop_fire;
  stop_fire.data = false;
  fire_pub->publish(stop_fire);
  rclcpp::shutdown();
  ros_spin_thread.join();
  cv::destroyAllWindows();
  tools::logger()->info("Program terminated successfully.");
  return 0;
}
