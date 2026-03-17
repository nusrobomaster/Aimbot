#include <fmt/core.h>
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <iomanip>
#include <iostream>
#include <unistd.h>
#include <thread>

#include "io/camera.hpp"
#include "io/usb_communication.hpp"
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

// ---------------------------------------------------------------------------
// Pipeline shared data structures
// ---------------------------------------------------------------------------

struct CameraFrame {
    cv::Mat img;
    steady_clock::time_point t;
    uint64_t frame_id = 0;
};

struct InferenceResult {
    cv::Mat img;
    std::list<auto_aim::Armor> armors;
    std::list<auto_aim::Target> targets;
    io::Command command;
    Eigen::Vector3d ypr;
    auto_aim::AimPoint aim_point;
    bool should_shoot = false;
    float cmd_yaw   = 0.0f;    // yaw delta to send (rad)
    float cmd_pitch = 0.0f;    // pitch delta to send (rad)
    uint64_t frame_id = 0;
};

/**
 * Thread-safe single-slot buffer.
 * Producer always overwrites (never blocks).
 * Consumer blocks until new data arrives or timeout.
 */
template<typename T>
class LatestSlot {
public:
    void push(T&& item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            slot_ = std::move(item);
            has_data_ = true;
        }
        cv_.notify_all();
    }

    bool pop(T& out, std::chrono::milliseconds timeout = 33ms) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, timeout, [this] { return has_data_.load(); }))
            return false;
        out = slot_;
        has_data_ = false;
        return true;
    }

    bool try_pop(T& out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!has_data_) return false;
        out = slot_;
        has_data_ = false;
        return true;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> has_data_{false};
    T slot_;
};

// ---------------------------------------------------------------------------
// Default config
// ---------------------------------------------------------------------------
const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/standard_onnx0526.yaml | 位置参数，yaml配置文件路径 }"
  "{serial         | /dev/ttyACM0 | STM32 CDC serial device path }";

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------
void drawArmorPlate(cv::Mat& img, const auto_aim::Armor& armor, const cv::Scalar& color,
                   const std::string& label_prefix = "") {
    if (armor.points.size() >= 4) {
        for (int i = 0; i < 4; i++)
            cv::line(img, armor.points[i], armor.points[(i + 1) % 4], color, 2);
        for (const auto& point : armor.points)
            cv::circle(img, point, 3, color, -1);
        cv::Point2f center(0, 0);
        for (const auto& point : armor.points) { center.x += point.x; center.y += point.y; }
        center.x /= 4; center.y /= 4;
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
    } else if (armor.box.area() > 0) {
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
        cv::Scalar armor_color = (armor_count == 0) ? cv::Scalar(0,255,0)
                               : (armor_count == 1) ? cv::Scalar(255,255,0) : color;
        for (int i = 0; i < 4; i++)
            cv::line(img, image_points[i], image_points[(i+1)%4], armor_color, 2);
        for (const auto& point : image_points) cv::circle(img, point, 3, armor_color, -1);
        cv::Point2f center(0,0);
        for (const auto& point : image_points) { center.x += point.x; center.y += point.y; }
        center.x /= 4; center.y /= 4;
        cv::circle(img, center, 4, armor_color, -1);
        armor_count++;
    }
    if (!armor_xyza_list.empty()) {
        auto first_armor_points = solver.reproject_armor(
            armor_xyza_list[0].head(3), armor_xyza_list[0][3], target.armor_type, target.name);
        cv::Point2f text_pos(0,0);
        for (const auto& point : first_armor_points) { text_pos.x += point.x; text_pos.y += point.y; }
        text_pos.x /= 4; text_pos.y /= 4;
        std::string target_info = fmt::format("{} (ID:{})", auto_aim::ARMOR_NAMES[target.name], target.last_id);
        cv::putText(img, target_info, cv::Point(text_pos.x+15, text_pos.y),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        Eigen::VectorXd x = target.ekf_x();
        if (x.size() >= 11) {
            double distance = std::sqrt(x[0]*x[0] + x[2]*x[2] + x[4]*x[4]);
            std::string state_text = fmt::format("Dist:{:.1f}m v:{:.1f}m/s", distance,
                Eigen::Vector3d(x[1],x[3],x[5]).norm());
            cv::putText(img, state_text, cv::Point(text_pos.x+15, text_pos.y+20),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200,200,200), 1);
        }
    }
}

void renderResult(cv::Mat& display_img, const InferenceResult& res,
                  auto_aim::Solver& solver, float fps,
                  const calibur::BoardStatus& board_status,
                  float display_scale = 0.75f) {
    int armor_idx = 0;
    for (const auto& armor : res.armors) {
        cv::Scalar color;
        switch (armor.color) {
            case auto_aim::Color::red:        color = cv::Scalar(0,0,255);   break;
            case auto_aim::Color::blue:       color = cv::Scalar(255,0,0);   break;
            case auto_aim::Color::purple:     color = cv::Scalar(255,0,255); break;
            case auto_aim::Color::extinguish: color = cv::Scalar(0,255,255); break;
            default:                          color = cv::Scalar(200,200,200);
        }
        drawArmorPlate(display_img, armor, color, fmt::format("D{}", armor_idx++));
    }
    for (const auto& target : res.targets) {
        cv::Scalar color;
        switch (target.name) {
            case auto_aim::ArmorName::outpost:
            case auto_aim::ArmorName::sentry: color = cv::Scalar(0,165,255); break;
            case auto_aim::ArmorName::base:   color = cv::Scalar(255,0,255); break;
            default:                          color = cv::Scalar(0,255,255);
        }
        drawTarget(display_img, target, solver, color);
    }
    if (res.aim_point.valid) {
        auto image_points = solver.reproject_armor(res.aim_point.xyza.head(3), res.aim_point.xyza[3],
            auto_aim::ArmorType::small, auto_aim::ArmorName::one);
        for (int i = 0; i < 4; i++)
            cv::line(display_img, image_points[i], image_points[(i+1)%4], cv::Scalar(0,0,255), 2);
        cv::Point2f center(0,0);
        for (const auto& point : image_points) { center.x += point.x; center.y += point.y; }
        center.x /= 4; center.y /= 4;
        cv::circle(display_img, center, 8, cv::Scalar(0,0,255), -1);
        cv::circle(display_img, center, 12, cv::Scalar(0,0,255), 2);
        cv::putText(display_img, "AIM", cv::Point(center.x+15, center.y),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);
    }
    cv::Point center(display_img.cols/2, display_img.rows/2);
    cv::line(display_img, cv::Point(center.x-20, center.y), cv::Point(center.x+20, center.y), cv::Scalar(0,255,0), 2);
    cv::line(display_img, cv::Point(center.x, center.y-20), cv::Point(center.x, center.y+20), cv::Scalar(0,255,0), 2);
    cv::circle(display_img, center, 8, cv::Scalar(0,255,0), 1);

    // Top-left: FPS + detection info
    cv::putText(display_img, fmt::format("FPS: {:.1f}  Armors:{}  Targets:{}", fps, res.armors.size(), res.targets.size()),
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,255), 2);
    cv::putText(display_img, fmt::format("Yaw:{:.1f}deg  Pitch:{:.1f}deg", res.ypr[0]*180.0/M_PI, res.ypr[1]*180.0/M_PI),
                cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200,200,100), 1);

    // Top-right: Board status
    int rx = display_img.cols - 320;
    cv::putText(display_img, fmt::format("Robot:{} HP:{}", board_status.robot_id, board_status.current_hp),
                cv::Point(rx, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(100,255,100), 1);
    cv::putText(display_img, fmt::format("Game:{} Time:{}", board_status.game_progress, board_status.time_left),
                cv::Point(rx, 55), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200,200,200), 1);
    cv::putText(display_img, fmt::format("Board Yaw:{:.1f} Pit:{:.1f}",
                board_status.yaw_angle, board_status.pitch_angle),
                cv::Point(rx, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(180,180,255), 1);

    // Bottom: delta command info (what is actually sent to the board)
    cv::putText(display_img, fmt::format("Delta: Y{:.2f}deg P{:.2f}deg", res.cmd_yaw*180.0/M_PI, res.cmd_pitch*180.0/M_PI),
                cv::Point(10, display_img.rows-60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,255,0), 2);
    if (res.command.control)
        cv::putText(display_img, "CONTROL ACTIVE", cv::Point(10, display_img.rows-90),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,165,0), 2);
    if (res.should_shoot)
        cv::putText(display_img, "FIRE!", cv::Point(10, display_img.rows-30),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,0,255), 3);
    if (res.command.control && (std::abs(res.cmd_yaw)>0.001 || std::abs(res.cmd_pitch)>0.001)) {
        int aim_x = center.x + static_cast<int>(res.cmd_yaw*180.0/M_PI*10);
        int aim_y = center.y - static_cast<int>(res.cmd_pitch*180.0/M_PI*10);
        cv::arrowedLine(display_img, center, cv::Point(aim_x, aim_y), cv::Scalar(255,0,255), 2, 8, 0, 0.1);
    }
    if (std::abs(display_scale - 1.0f) > 0.01f)
        cv::resize(display_img, display_img, cv::Size(), display_scale, display_scale);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    cv::CommandLineParser cli(argc, argv, keys);
    auto config_path = cli.get<std::string>(0);
    auto serial_path = cli.get<std::string>("serial");
    if (cli.has("help") || config_path.empty()) {
        cli.printMessage();
        return 0;
    }

    tools::Exiter exiter;

    // ── USB CDC to STM32 ─────────────────────────────────────────────────
    calibur::USBCommunication usb(serial_path);
    if (!usb.open()) {
        tools::logger()->error("Failed to open USB device at {}. Continuing without board.", serial_path);
    } else {
        tools::logger()->info("[USB] Connected to {}", serial_path);
    }

    // ── Camera & IMU ─────────────────────────────────────────────────────
    io::Camera camera(config_path);
    H30IMU imu("/dev/ttyACM1", 460800);
    if (!imu.start()) tools::logger()->warn("Failed to start external IMU");

    // ── Vision pipeline ──────────────────────────────────────────────────
    auto_aim::YOLO     detector(config_path, false);
    auto_aim::Solver   solver(config_path);
    auto_aim::Tracker  tracker(config_path, solver);
    auto_aim::Aimer    aimer(config_path);
    auto_aim::Shooter  shooter(config_path);

    // ── Command smoothing (tunable from YAML) ────────────────────────────
    auto yaml_cfg = YAML::LoadFile(config_path);
    const float cmd_alpha     = yaml_cfg["cmd_smooth_alpha"] ? yaml_cfg["cmd_smooth_alpha"].as<float>() : 0.25f;
    const float cmd_dead_zone = yaml_cfg["cmd_dead_zone"]    ? yaml_cfg["cmd_dead_zone"].as<float>()    : 0.0005f;
    tools::logger()->info("[Smoothing] alpha={:.2f}  dead_zone={:.4f} rad", cmd_alpha, cmd_dead_zone);

    // ── IMU tare ─────────────────────────────────────────────────────────
    std::this_thread::sleep_for(500ms);
    Eigen::Quaterniond q_zero = Eigen::Quaterniond::Identity();
    {
        Eigen::Vector4d q_sum = Eigen::Vector4d::Zero();
        for (int i = 0; i < 100; ++i) {
            auto tp = steady_clock::now();
            Eigen::Quaterniond qs = imu.getQuaternionAt(tp);
            Eigen::Vector4d c = qs.coeffs();
            if (i > 0 && c.dot(q_sum) < 0) c = -c;
            q_sum += c;
            std::this_thread::sleep_for(5ms);
        }
        q_zero.coeffs() = q_sum.normalized();
    }
    tools::logger()->info("IMU tare complete.");

    // =========================================================
    // Pipeline slots
    // =========================================================
    LatestSlot<CameraFrame>     cam_slot;
    LatestSlot<InferenceResult> publish_slot;   // Inference → USB TX (latency-critical)
    LatestSlot<InferenceResult> display_slot;   // Inference → Display (best-effort)

    std::atomic<bool> pipeline_running{true};

    // =========================================================
    // Thread 1: Camera Capture
    // =========================================================
    std::thread cam_thread([&]() {
        tools::logger()->info("[CamThread] Started.");
        uint64_t fid = 0;
        while (pipeline_running && !exiter.exit()) {
            CameraFrame frame;
            camera.read(frame.img, frame.t);
            frame.frame_id = fid++;
            cam_slot.push(std::move(frame));
        }
        tools::logger()->info("[CamThread] Stopped.");
    });

    // =========================================================
    // Thread 2: Inference (Critical Path)
    // =========================================================
    std::thread infer_thread([&]() {
        tools::logger()->info("[InferThread] Started.");

        float t_imu_ms = 0, t_det_ms = 0, t_trk_ms = 0, t_aim_ms = 0;
        uint64_t iter = 0;

        while (pipeline_running && !exiter.exit()) {
            CameraFrame frame;
            if (!cam_slot.pop(frame, 50ms)) continue;

            auto t0 = steady_clock::now();

            // IMU
            Eigen::Quaterniond q_raw   = imu.getQuaternionAt(frame.t - 1ms);
            Eigen::Quaterniond q_tared = q_zero.conjugate() * q_raw;
            solver.set_R_gimbal2world(q_tared);
            Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

            auto t1 = steady_clock::now();

            // Detection
            auto armors = detector.detect(frame.img);

            auto t2 = steady_clock::now();

            // Tracking
            auto targets = tracker.track(armors, frame.t, true);

            auto t3 = steady_clock::now();

            // Aiming
            auto command = aimer.aim(targets, frame.t, 23.0);
            bool should_shoot = false;
            if (!targets.empty()) {
                Eigen::Vector3d gimbal_pos(command.yaw, command.pitch, 0);
                should_shoot = shooter.shoot(command, aimer, targets, gimbal_pos);
            }

            auto t4 = steady_clock::now();

            // Compute yaw/pitch deltas to send to board
            float cmd_yaw   = 0.0f;
            float cmd_pitch = 0.0f;
            if (aimer.debug_aim_point.valid && !targets.empty()) {
                cmd_yaw   = static_cast<float>(-tools::limit_rad(command.yaw - ypr[0]));
                cmd_pitch = static_cast<float>(-(command.pitch + ypr[1]));
            }

            // Build result
            InferenceResult res;
            res.img          = frame.img;
            res.armors       = armors;
            res.targets      = targets;
            res.command      = command;
            res.ypr          = ypr;
            res.aim_point    = aimer.debug_aim_point;
            res.should_shoot = should_shoot;
            res.cmd_yaw      = cmd_yaw;
            res.cmd_pitch    = cmd_pitch;
            res.frame_id     = frame.frame_id;

            // Push to publish slot first (latency-critical)
            publish_slot.push(InferenceResult(res));

            // Push to display slot (best-effort)
            display_slot.push(std::move(res));

            // Timing log + debug
            t_imu_ms += duration<float,std::milli>(t1-t0).count();
            t_det_ms += duration<float,std::milli>(t2-t1).count();
            t_trk_ms += duration<float,std::milli>(t3-t2).count();
            t_aim_ms += duration<float,std::milli>(t4-t3).count();
            if (++iter % 30 == 0) {
                tools::logger()->info(
                    "[Timing] imu={:.1f}ms det={:.1f}ms trk={:.1f}ms aim={:.1f}ms  total={:.1f}ms",
                    t_imu_ms/30, t_det_ms/30, t_trk_ms/30, t_aim_ms/30,
                    (t_imu_ms+t_det_ms+t_trk_ms+t_aim_ms)/30);
                t_imu_ms = t_det_ms = t_trk_ms = t_aim_ms = 0;
                for (const auto& a : armors) {
                    tools::logger()->info(
                        "[Det] {} ({}) conf={:.2f} center=({:.0f},{:.0f}) world=({:.2f},{:.2f},{:.2f})",
                        auto_aim::ARMOR_NAMES[a.name], auto_aim::ARMOR_TYPES[a.type],
                        a.confidence, a.center.x, a.center.y,
                        a.xyz_in_world[0], a.xyz_in_world[1], a.xyz_in_world[2]);
                }
                if (!targets.empty()) {
                    tools::logger()->info(
                        "[Trk] name={} cmd_yaw={:.1f}deg cmd_pitch={:.1f}deg delta_yaw={:.2f}deg delta_pitch={:.2f}deg",
                        auto_aim::ARMOR_NAMES[targets.front().name],
                        command.yaw*180.0/M_PI, command.pitch*180.0/M_PI,
                        cmd_yaw*180.0/M_PI, cmd_pitch*180.0/M_PI);
                }
            }
        }
        tools::logger()->info("[InferThread] Stopped.");
    });

    // =========================================================
    // Thread 3: USB Publish (replaces ROS publish thread)
    // =========================================================
    bool prev_fire_state = false;
    std::thread publish_thread([&]() {
        tools::logger()->info("[USBPubThread] Started.");

        float smooth_yaw   = 0.0f;
        float smooth_pitch = 0.0f;

        InferenceResult res;
        while (pipeline_running && !exiter.exit()) {
            if (!publish_slot.pop(res, 50ms)) continue;

            float send_yaw   = 0.0f;
            float send_pitch = 0.0f;

            if (res.command.control) {
                smooth_yaw   = cmd_alpha * res.cmd_yaw   + (1.0f - cmd_alpha) * smooth_yaw;
                smooth_pitch = cmd_alpha * res.cmd_pitch  + (1.0f - cmd_alpha) * smooth_pitch;

                send_yaw   = smooth_yaw;
                send_pitch = smooth_pitch;

                if (std::abs(send_yaw)   < cmd_dead_zone) send_yaw   = 0.0f;
                if (std::abs(send_pitch) < cmd_dead_zone) send_pitch = 0.0f;
            } else {
                smooth_yaw   = 0.0f;
                smooth_pitch = 0.0f;
            }

            usb.sendGimbalCommand(send_yaw, send_pitch);

            // Send fire command on state change only
            if (res.should_shoot != prev_fire_state) {
                usb.sendFiringCommand(res.should_shoot);
                prev_fire_state = res.should_shoot;
            }
        }
        tools::logger()->info("[USBPubThread] Stopped.");
    });

    // =========================================================
    // Thread 4 (Main): Display
    // =========================================================
    tools::logger()->info("[DisplayThread] Started (main thread).");
    cv::namedWindow("Auto Aim Results", cv::WINDOW_NORMAL);
    cv::resizeWindow("Auto Aim Results", 1280, 720);

    auto  last_fps_time = steady_clock::now();
    int   display_frame_count = 0;
    float fps = 0.0f;
    cv::Mat display_buf;
    int save_count = 0;

    while (!exiter.exit()) {
        InferenceResult res;
        if (!display_slot.pop(res, 100ms)) {
            cv::waitKey(1);
            continue;
        }

        // Get latest board status for overlay
        auto board_status = usb.getStatus();

        res.img.copyTo(display_buf);
        renderResult(display_buf, res, solver, fps, board_status);
        cv::imshow("Auto Aim Results", display_buf);

        char key = cv::waitKey(1);
        if (key == 27) {
            tools::logger()->info("ESC pressed — shutting down.");
            break;
        } else if (key == 's' || key == 'S') {
            cv::imwrite(fmt::format("frame_{:04d}.png", save_count++), display_buf);
            tools::logger()->info("Saved frame_{:04d}.png", save_count-1);
        } else if (key == 'p' || key == 'P') {
            tools::logger()->info("Paused. Press any key to continue.");
            cv::waitKey(0);
        } else if (key == 'd' || key == 'D') {
            tools::logger()->info(
                "Armors:{} Targets:{} Cmd yaw={:.3f}deg pitch={:.3f}deg shoot={} | "
                "Board HP:{} RxPkt:{} CRC_fail:{}",
                res.armors.size(), res.targets.size(),
                res.command.yaw*180.0/M_PI, res.command.pitch*180.0/M_PI,
                res.should_shoot,
                board_status.current_hp, usb.rxPacketCount(), usb.rxCrcFailCount());
        } else if (key == 'c' || key == 'C') {
            system("clear");
        }

        // FPS counter
        display_frame_count++;
        if (display_frame_count % 30 == 0) {
            auto now     = steady_clock::now();
            auto elapsed = duration_cast<milliseconds>(now - last_fps_time).count();
            fps = 30000.0f / static_cast<float>(elapsed);
            tools::logger()->info("[Display] FPS: {:.1f} | USB RX:{} pkts, CRC_fail:{}, TX:{}",
                                  fps, usb.rxPacketCount(), usb.rxCrcFailCount(), usb.txPacketCount());
            last_fps_time = now;
        }
    }

    // =========================================================
    // Shutdown
    // =========================================================
    tools::logger()->info("Shutting down pipeline...");
    pipeline_running = false;

    // Send zero commands before closing
    usb.sendGimbalCommand(0.0f, 0.0f);
    usb.sendFiringCommand(false);

    cam_thread.join();
    infer_thread.join();
    publish_thread.join();

    usb.close();

    cv::destroyAllWindows();
    tools::logger()->info("Program terminated successfully.");
    return 0;
}