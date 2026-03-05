#include <fmt/core.h>
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>
#include <thread>
#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "io/h30_imu.hpp"  
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/multithread/usb_communication.h"
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

const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/standard3.yaml | 位置参数，yaml配置文件路径 }";

/**
 * @brief Draw armor plate with its keypoints and information
 * 
 * @param img Image to draw on
 * @param armor Armor plate to visualize
 * @param color Color to use for drawing
 * @param label_prefix Optional label prefix
 */
void drawArmorPlate(cv::Mat& img, const auto_aim::Armor& armor, const cv::Scalar& color, 
                   const std::string& label_prefix = "") {
    // Draw the armor using its points (4 corners)
    if (armor.points.size() >= 4) {
        // Draw armor polygon (quadrilateral)
        for (int i = 0; i < 4; i++) {
            cv::line(img, armor.points[i], armor.points[(i + 1) % 4], color, 2);
        }
        
        // Draw corners
        for (const auto& point : armor.points) {
            cv::circle(img, point, 3, color, -1);
        }
        
        // Draw armor center (the actual center from the points)
        cv::Point2f center(0, 0);
        for (const auto& point : armor.points) {
            center.x += point.x;
            center.y += point.y;
        }
        center.x /= 4;
        center.y /= 4;
        cv::circle(img, center, 4, cv::Scalar(255, 255, 255), -1);
        
        // Draw label with information
        std::string label;
        if (!label_prefix.empty()) {
            label = label_prefix + " ";
        }
        
        // Add armor name
        label += auto_aim::ARMOR_NAMES[armor.name] + " ";
        
        // Add armor type
        label += "(" + auto_aim::ARMOR_TYPES[armor.type] + ") ";
        
        // Add confidence
        label += fmt::format("{:.2f}", armor.confidence);
        
        cv::putText(img, label, cv::Point(center.x + 10, center.y),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        
        // Draw 3D position if available
        if (armor.xyz_in_world.norm() > 0) {
            std::string pos_text = fmt::format("({:.2f},{:.2f},{:.2f})m", 
                                              armor.xyz_in_world[0],
                                              armor.xyz_in_world[1],
                                              armor.xyz_in_world[2]);
            cv::putText(img, pos_text, cv::Point(center.x + 10, center.y + 20),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
        }
        
        // Draw lightbars
        cv::line(img, armor.left.top, armor.left.bottom, cv::Scalar(0, 255, 255), 2);
        cv::line(img, armor.right.top, armor.right.bottom, cv::Scalar(0, 255, 255), 2);
    }
    // Fallback: draw bounding box if points not available
    else if (armor.box.area() > 0) {
        cv::rectangle(img, armor.box, color, 2);
        cv::Point center(armor.box.x + armor.box.width/2, armor.box.y + armor.box.height/2);
        cv::circle(img, center, 3, color, -1);
        
        std::string label = label_prefix + " " + auto_aim::ARMOR_NAMES[armor.name];
        cv::putText(img, label, cv::Point(armor.box.x, armor.box.y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
}

/**
 * @brief Draw target with all its armor plates and trajectory
 * 
 * @param img Image to draw on
 * @param target Target to visualize
 * @param solver Solver for reprojection
 * @param color Color to use for drawing
 */
void drawTarget(cv::Mat& img, const auto_aim::Target& target, 
                auto_aim::Solver& solver, const cv::Scalar& color) {
    // Draw all armor plates of this target using reprojection
    auto armor_xyza_list = target.armor_xyza_list();
    int armor_count = 0;
    
    for (const Eigen::Vector4d& xyza : armor_xyza_list) {
        // Get armor corners from 3D position
        auto image_points = solver.reproject_armor(
            xyza.head(3), xyza[3], target.armor_type, target.name);
        
        // Draw this armor plate
        cv::Scalar armor_color = color;
        if (armor_count == 0) armor_color = cv::Scalar(0, 255, 0);  // Green for first
        else if (armor_count == 1) armor_color = cv::Scalar(255, 255, 0);  // Yellow for second
        
        // Draw armor quadrilateral
        for (int i = 0; i < 4; i++) {
            cv::line(img, image_points[i], image_points[(i + 1) % 4], armor_color, 2);
        }
        
        // Draw corners
        for (const auto& point : image_points) {
            cv::circle(img, point, 3, armor_color, -1);
        }
        
        // Draw center
        cv::Point2f center(0, 0);
        for (const auto& point : image_points) {
            center.x += point.x;
            center.y += point.y;
        }
        center.x /= 4;
        center.y /= 4;
        cv::circle(img, center, 4, armor_color, -1);
        
        armor_count++;
    }
    
    // Draw target information
    if (!armor_xyza_list.empty()) {
        // Get center of first armor for text placement
        auto first_armor_points = solver.reproject_armor(
            armor_xyza_list[0].head(3), armor_xyza_list[0][3], 
            target.armor_type, target.name);
        cv::Point2f text_pos(0, 0);
        for (const auto& point : first_armor_points) {
            text_pos.x += point.x;
            text_pos.y += point.y;
        }
        text_pos.x /= 4;
        text_pos.y /= 4;
        
        // Draw target ID and armor name
        std::string target_info = fmt::format("{} (ID:{})", 
                                             auto_aim::ARMOR_NAMES[target.name],
                                             target.last_id);
        cv::putText(img, target_info, 
                   cv::Point(text_pos.x + 15, text_pos.y),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        
        // Draw target state if available
        Eigen::VectorXd x = target.ekf_x();
        if (x.size() >= 11) {
            double distance = std::sqrt(x[0]*x[0] + x[2]*x[2] + x[4]*x[4]);
            std::string state_text = fmt::format("Dist: {:.1f}m v:{:.1f}m/s", 
                                                distance,
                                                Eigen::Vector3d(x[1], x[3], x[5]).norm());
            cv::putText(img, state_text, 
                       cv::Point(text_pos.x + 15, text_pos.y + 20),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
        }
    }
}

/**
 * @brief Main display function to visualize detection, tracking and aiming results
 * 
 * @param img Input image
 * @param armors Detected armors from YOLO
 * @param targets Tracked targets
 * @param solver Solver for 3D calculations
 * @param command Aiming command
 * @param ypr Yaw-Pitch-Roll angles
 * @param aim_point Current aim point
 * @param display_scale Scale factor for display window
 */
void displayAutoAimResults(cv::Mat& img, 
                          const std::list<auto_aim::Armor>& armors,
                          const std::list<auto_aim::Target>& targets,
                          auto_aim::Solver& solver,
                          const io::Command& command,
                          const Eigen::Vector3d& ypr,
                          const auto_aim::AimPoint& aim_point,
                          float display_scale = 0.75f) {
    
    cv::Mat display_img = img.clone();
    
    // 1. Display basic information on top
    std::string info_text = fmt::format("Armors: {} | Targets: {}", 
                                        armors.size(), targets.size());
    cv::putText(display_img, info_text, cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    
    std::string ypr_text = fmt::format("Yaw: {:.1f}° | Pitch: {:.1f}°", 
                                       ypr[0], ypr[1]);
    cv::putText(display_img, ypr_text, cv::Point(10, 60), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 100), 1);
    
    // 2. Draw detected armors (different colors based on armor type/color)
    int armor_idx = 0;
    for (const auto& armor : armors) {
        cv::Scalar color;
        
        // Color based on armor color
        switch (armor.color) {
            case auto_aim::Color::red:
                color = cv::Scalar(0, 0, 255);  // Red in BGR
                break;
            case auto_aim::Color::blue:
                color = cv::Scalar(255, 0, 0);  // Blue in BGR
                break;
            case auto_aim::Color::purple:
                color = cv::Scalar(255, 0, 255);  // Purple in BGR
                break;
            case auto_aim::Color::extinguish:
                color = cv::Scalar(0, 255, 255);  // Cyan for extinguish
                break;
            default:
                color = cv::Scalar(200, 200, 200);  // Gray
        }
        
        // Make small armors brighter
        if (armor.type == auto_aim::ArmorType::small) {
            color = cv::Scalar(color[0] * 1.2, color[1] * 1.2, color[2] * 1.2);
        }
        
        std::string label = fmt::format("D{}", armor_idx);
        drawArmorPlate(display_img, armor, color, label);
        armor_idx++;
    }
    
    // 3. Draw tracked targets (yellow color for tracking)
    int target_idx = 0;
    for (const auto& target : targets) {
        cv::Scalar color;
        
        // Different colors for different target types
        switch (target.name) {
            case auto_aim::ArmorName::outpost:
            case auto_aim::ArmorName::sentry:
                color = cv::Scalar(0, 165, 255);  // Orange for special targets
                break;
            case auto_aim::ArmorName::base:
                color = cv::Scalar(255, 0, 255);  // Magenta for base
                break;
            default:
                color = cv::Scalar(0, 255, 255);  // Yellow for normal targets
        }
        
        drawTarget(display_img, target, solver, color);
        target_idx++;
    }
    
    // 4. Draw aiming point (red color)
    if (aim_point.valid) {
        Eigen::Vector4d aim_xyza = aim_point.xyza;
        // Use a default armor type and name for aim point visualization
        auto image_points = solver.reproject_armor(
            aim_xyza.head(3), aim_xyza[3], 
            auto_aim::ArmorType::small,  // Default type
            auto_aim::ArmorName::one);   // Default name
        
        // Draw aiming point as a red rectangle
        for (int i = 0; i < 4; i++) {
            cv::line(display_img, image_points[i], image_points[(i + 1) % 4], 
                    cv::Scalar(0, 0, 255), 2);
        }
        
        // Draw center as red circle
        cv::Point2f center(0, 0);
        for (const auto& point : image_points) {
            center.x += point.x;
            center.y += point.y;
        }
        center.x /= 4;
        center.y /= 4;
        cv::circle(display_img, center, 8, cv::Scalar(0, 0, 255), -1);
        cv::circle(display_img, center, 12, cv::Scalar(0, 0, 255), 2);
        
        // Draw "AIM" label
        cv::putText(display_img, "AIM", cv::Point(center.x + 15, center.y),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    }
    
    // 5. Draw crosshair (green) at image center
    cv::Point center(display_img.cols / 2, display_img.rows / 2);
    cv::line(display_img, cv::Point(center.x - 20, center.y), 
             cv::Point(center.x + 20, center.y), cv::Scalar(0, 255, 0), 2);
    cv::line(display_img, cv::Point(center.x, center.y - 20), 
             cv::Point(center.x, center.y + 20), cv::Scalar(0, 255, 0), 2);
    cv::circle(display_img, center, 8, cv::Scalar(0, 255, 0), 1);
    
    // 6. Draw command information
    std::string cmd_text = fmt::format("Cmd: Y{:.2f}° P{:.2f}°", 
                                       command.yaw , command.pitch);
    cv::putText(display_img, cmd_text, cv::Point(10, display_img.rows - 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
    
    if (command.control) {
        std::string control_text = "CONTROL ACTIVE";
        cv::putText(display_img, control_text, cv::Point(10, display_img.rows - 90),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 165, 0), 2);
    }
    
    if (command.shoot) {
        cv::putText(display_img, "FIRE!", cv::Point(10, display_img.rows - 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 3);
    }
    
    // 7. Draw predicted aim direction from command
    if (command.control && (std::abs(command.yaw) > 0.001 || std::abs(command.pitch) > 0.001)) {
        int aim_x = center.x + static_cast<int>(command.yaw * 500);  // Scaled for visibility
        int aim_y = center.y + static_cast<int>(command.pitch * 500);
        cv::arrowedLine(display_img, center, cv::Point(aim_x, aim_y), 
                       cv::Scalar(255, 0, 255), 2, 8, 0, 0.1);
    }
    
    // 8. Draw color legend
    int legend_y = 90;
    int legend_step = 25;
    
    cv::putText(display_img, "Legend:", cv::Point(10, legend_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    cv::putText(display_img, "Red Armor", cv::Point(30, legend_y + legend_step),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
    cv::putText(display_img, "Blue Armor", cv::Point(30, legend_y + legend_step * 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);
    cv::putText(display_img, "Target", cv::Point(30, legend_y + legend_step * 3),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 255), 1);
    cv::putText(display_img, "Aim Point", cv::Point(30, legend_y + legend_step * 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
    
    // 9. Resize for display if needed
    if (std::abs(display_scale - 1.0f) > 0.01f) {
        cv::resize(display_img, display_img, 
                  cv::Size(), display_scale, display_scale);
    }
    
    // 10. Show image
    cv::imshow("Auto Aim Results", display_img);
    
    // Handle keyboard input
    char key = cv::waitKey(1);
    if (key == 27) { // ESC key
        throw std::runtime_error("User pressed ESC to exit");
    }
    else if (key == 's' || key == 'S') { // Save current frame
        static int save_count = 0;
        std::string filename = fmt::format("frame_{:04d}.png", save_count++);
        cv::imwrite(filename, display_img);
        tools::logger()->info("Saved frame to {}", filename);
    }
    else if (key == 'p' || key == 'P') { // Pause
        tools::logger()->info("Paused. Press any key to continue...");
        cv::waitKey(0);
    }
    else if (key == 'd' || key == 'D') { // Debug info
        tools::logger()->info("=== Debug Info ===");
        tools::logger()->info("Armors detected: {}", armors.size());
        tools::logger()->info("Targets tracked: {}", targets.size());
        tools::logger()->info("Command: yaw={:.3f}°, pitch={:.3f}°, shoot={}, control={}", 
                             command.yaw, command.pitch, 
                             command.shoot, command.control);
        tools::logger()->info("Gimbal YPR: [{:.1f}°, {:.1f}°, {:.1f}°]", 
                             ypr[0], ypr[1], ypr[2]);
        tools::logger()->info("Aim Point valid: {}", aim_point.valid);
        
        // List all armors
        int i = 0;
        for (const auto& armor : armors) {
            tools::logger()->info("Armor {}: {} ({}) conf:{:.2f} pos:({:.2f},{:.2f},{:.2f})",
                                 i++, auto_aim::ARMOR_NAMES[armor.name],
                                 auto_aim::ARMOR_TYPES[armor.type],
                                 armor.confidence,
                                 armor.xyz_in_world[0], armor.xyz_in_world[1], armor.xyz_in_world[2]);
        }
        
        // List all targets
        i = 0;
        for (const auto& target : targets) {
            tools::logger()->info("Target {}: {} ID:{}",
                                 i++, auto_aim::ARMOR_NAMES[target.name],
                                 target.last_id);
        }
    }
    else if (key == '1') { // Toggle armor display
        static bool show_armors = true;
        show_armors = !show_armors;
        tools::logger()->info("Show armors: {}", show_armors);
    }
    else if (key == '2') { // Toggle target display
        static bool show_targets = true;
        show_targets = !show_targets;
        tools::logger()->info("Show targets: {}", show_targets);
    }
    else if (key == 'c' || key == 'C') { // Clear log
        system("clear");  // For Linux/Mac, use "cls" for Windows
    }
}

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;

  io::Camera camera(config_path);
  // io::CBoard cboard(config_path); // Uncomment if using serial communication
  H30IMU imu("/dev/ttyACM0", 460800);
  if (!imu.start()) {
    tools::logger()->warn("Failed to start external IMU");
}
  auto_aim::YOLO detector(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);

  // ============ USB COMMUNICATION SETUP ============
  calibur::USBCommunication usb_comm;

  std::string usb_port = "/dev/ttyACM1";
  int baudrate = 115200;  // Match your MCU baud rate
  
  std::cout << "\n[USB] Attempting to open " << usb_port << " at " << baudrate << " baud" << std::endl;
  
  // Check if device exists
  if (access(usb_port.c_str(), F_OK) != 0) {
    std::cerr << "[USB] ERROR: " << usb_port << " does not exist!" << std::endl;
    std::cerr << "[USB] Run: ls -la /dev/ttyACM* to see available devices" << std::endl;
  } 
  // Check permissions
  else if (access(usb_port.c_str(), R_OK | W_OK) != 0) {
    std::cerr << "[USB] ERROR: No read/write permission for " << usb_port << std::endl;
    std::cerr << "[USB] Fix: sudo chmod 666 " << usb_port << std::endl;
  }
  // Try to open
  else if (!usb_comm.open(usb_port, baudrate)) {
    std::cerr << "[USB] ERROR: Failed to open " << usb_port << std::endl;
    std::cerr << "[USB] Check if another program is using it: sudo lsof " << usb_port << std::endl;
  } else {
    std::cout << "[USB] Successfully connected to " << usb_port << std::endl;
    usb_comm.flush();  // Clear any stale data
  }
  // =================================================

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;

  // Create display window
  cv::namedWindow("Auto Aim Results", cv::WINDOW_NORMAL);
  cv::resizeWindow("Auto Aim Results", 1280, 720);
  
  // For FPS calculation
  auto last_time = std::chrono::steady_clock::now();
  int frame_count = 0;
  float fps = 0.0f;

  // ============ USB TIMING AND STATISTICS ============
  auto last_usb_send = std::chrono::steady_clock::now();
  const auto usb_interval = std::chrono::milliseconds(10); // 100Hz send rate
  
  // Statistics
  int usb_send_count = 0;
  int usb_error_count = 0;
  int no_target_count = 0;
  
  // Track previous fire state (0 = stop, 1 = start)
  uint8_t prev_fire_state = 0;
  
  // For shooter debug
  int shooter_decision_count = 0;
  // ===================================================

  std::cout << "\n=== Auto-Aim Visualization System ===" << std::endl;
  std::cout << "Controls:" << std::endl;
  std::cout << "  ESC   - Exit program" << std::endl;
  std::cout << "  s/S   - Save current frame" << std::endl;
  std::cout << "  p/P   - Pause/continue" << std::endl;
  std::cout << "  d/D   - Show debug info" << std::endl;
  std::cout << "  1     - Toggle armor display" << std::endl;
  std::cout << "  2     - Toggle target display" << std::endl;
  std::cout << "  c/C   - Clear console" << std::endl;
  std::cout << "===================================\n" << std::endl;

  while (!exiter.exit()) {
    camera.read(img, t);
    // q = cboard.imu_at(t - 1ms); // Uncomment if using IMU
    Eigen::Quaterniond q = imu.getQuaternionAt(t - 1ms);
    
    // For testing, use identity quaternion if no IMU
    // if (q.coeffs().norm() == 0) {
    //     q = Eigen::Quaterniond::Identity();
    // }

    solver.set_R_gimbal2world(q);

    Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    auto armors = detector.detect(img);

    auto targets = tracker.track(armors, t, true);

    auto command = aimer.aim(targets, t, 23.0);
    
    // ============ SHOOTER DECISION ============
    bool should_shoot = false;
    
    if (!targets.empty() && usb_comm.isOpen()) {
        // Get gimbal position (current yaw/pitch from command)
        Eigen::Vector3d gimbal_pos(command.yaw, command.pitch, 0);
        
        // Shooter decides if we should actually fire based on:
        // - If we're on target (within tolerance)
        // - If command is stable
        // - If aim point is valid
        // - auto_fire = true (from config)
        should_shoot = shooter.shoot(command, aimer, targets, gimbal_pos);
        
        // Debug shooter decisions periodically
        shooter_decision_count++;
        if (shooter_decision_count % 50 == 0) {
            double distance = 0.0;
            if (!targets.empty() && !targets.begin()->armor_xyza_list().empty()) {
                distance = targets.begin()->armor_xyza_list()[0].head(3).norm();
            }
            std::cout << "[SHOOTER] Decision: " << (should_shoot ? "FIRE 🔫" : "HOLD ⏸️") 
                      << " | distance=" << std::fixed << std::setprecision(2) << distance << "m"
                      << " | targets=" << targets.size()
                      << " | aim_valid=" << aimer.debug_aim_point.valid
                      << std::endl;
        }
    }
    // ==========================================
    
    // ============ USB SEND AT 100Hz ============
    auto now = std::chrono::steady_clock::now();
    auto usb_elapsed = now - last_usb_send;
    
    if (usb_elapsed >= usb_interval) {
        bool gimbal_sent = false;
        
        if (usb_comm.isOpen()) {
            // Convert fire command to uint8_t (0 or 1) - USE SHOOTER'S DECISION
            uint8_t current_fire = should_shoot ? 1 : 0;
            
            // ALWAYS send gimbal command (either target angles or zero)
            if (aimer.debug_aim_point.valid && !targets.empty()) {
                // Valid target - send gimbal command with target angles
                gimbal_sent = usb_comm.sendGimbalCommand(
                    command.yaw,      // yaw in radians
                    command.pitch     // pitch in radians
                );
            } else {
                // No target - send zero gimbal command (gimbal stays still)
                gimbal_sent = usb_comm.sendGimbalCommand(0.0f, 0.0f);
            }
            
            // Send firing command ONLY if state changed
            if (current_fire != prev_fire_state) {
                bool fire_sent = usb_comm.sendFiringCommand(current_fire);
                
                if (fire_sent) {
                    prev_fire_state = current_fire;
                    
                    if (current_fire) {
                        std::cout << "[USB] 🔫 FIRE command sent! (value=" << (int)current_fire << ")" << std::endl;
                    } else {
                        std::cout << "[USB] 🔫 STOP FIRE command sent!" << std::endl;
                    }
                } else {
                    std::cout << "[USB] ❌ Failed to send firing command!" << std::endl;
                }
            }
            
            // Update statistics based on gimbal send
            if (gimbal_sent) {
                if (aimer.debug_aim_point.valid && !targets.empty()) {
                    usb_send_count++;
                    no_target_count = 0;
                    
                    // Print every 100th successful send
                    if (usb_send_count % 100 == 0) {
                        const auto& target = *targets.begin();
                        float distance = 0.0f;
                        if (!target.armor_xyza_list().empty()) {
                            distance = target.armor_xyza_list()[0].head(3).norm();
                        }
                        
                        std::cout << "[USB] Sent: yaw=" << command.yaw * 180.0f / M_PI 
                                 << "° pitch=" << command.pitch * 180.0f / M_PI
                                 << "° fire=" << (int)current_fire
                                 << " dist=" << distance << "m" << std::endl;
                    }
                } else {
                    no_target_count++;
                    
                    // Print every 100th frame with no target
                    if (no_target_count % 100 == 0) {
                        std::cout << "[USB] No target - zero command" << std::endl;
                    }
                }
            } else {
                usb_error_count++;
                if (usb_error_count % 100 == 0) {
                    std::cerr << "[USB] ❌ Failed to send gimbal command (" << usb_error_count << " errors)" << std::endl;
                    
                    // Try to reconnect if too many errors
                    if (usb_error_count > 1000) {
                        std::cerr << "[USB] Too many errors, attempting to reconnect..." << std::endl;
                        usb_comm.close();
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        usb_comm.open(usb_port, baudrate);
                        usb_error_count = 0;
                    }
                }
            }
            
            // Optional: Read any response from MCU (competition status, etc.)
            uint8_t rx_buffer[64];
            int bytes_read = usb_comm.receiveData(rx_buffer, sizeof(rx_buffer), 0);
            if (bytes_read > 0) {
                // Process incoming packets from MCU
                static int rx_count = 0;
                rx_count++;
                
                // Look for packets with magic byte 0x7F
                for (int i = 0; i < bytes_read - 1; i++) {
                    if (rx_buffer[i] == 0x7F) {  // USB_MAGIC_BYTE
                        uint8_t packet_id = rx_buffer[i + 1];
                        
                        // Print every 100th received packet
                        if (rx_count % 100 == 0) {
                            std::cout << "[USB] Received packet ID: " << (int)packet_id 
                                     << " from MCU" << std::endl;
                        }
                    }
                }
            }
        } else {
            // USB is not open - try to reconnect periodically
            static int reconnect_counter = 0;
            if (++reconnect_counter % 300 == 0) {  // Try every ~3 seconds at 100Hz
                std::cout << "[USB] Attempting to reconnect to " << usb_port << std::endl;
                usb_comm.open(usb_port, baudrate);
            }
        }
        
        last_usb_send = now;
    }
    // ============================================
    
    // Display results
    try {
        displayAutoAimResults(img, armors, targets, solver, command, 
                             ypr, aimer.debug_aim_point);
    } catch (const std::exception& e) {
        tools::logger()->error("Display error: {}", e.what());
        break;
    }
    
    // Calculate FPS
    frame_count++;
    if (frame_count % 30 == 0) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
        fps = 30000.0f / elapsed;
        last_time = now;
        if (frame_count % 100 == 0) {
            tools::logger()->info("Frame: {}, FPS: {:.1f}", frame_count, fps);
        }
    }
  }

  // ============ CLEANUP ============
  // Send zero commands before exiting
  if (usb_comm.isOpen()) {
    std::cout << "[USB] Shutting down - sending zero commands..." << std::endl;
    usb_comm.sendGimbalCommand(0.0f, 0.0f);
    usb_comm.sendFiringCommand(0);
    usb_comm.close();
  }
  // ================================

  cv::destroyAllWindows();
  tools::logger()->info("Program terminated successfully.");
  return 0;
}