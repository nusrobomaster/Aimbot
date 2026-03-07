/**
 * @file calibrate_imu.cpp
 * @brief Interactive IMU-to-gimbal rotation calibration tool
 *
 * Determines the correct R_gimbal2imubody matrix by having the user
 * perform two simple motions while viewing an armor plate:
 *   1. Yaw the gimbal ~30° to the right
 *   2. Pitch the gimbal ~20° upward
 *
 * The tool detects an armor plate (visual feedback + position verification),
 * records the IMU quaternion at each step, and computes which IMU body axes
 * correspond to gimbal yaw/pitch/forward, producing R_gimbal2imubody.
 *
 * Usage:
 *   ./build/calibrate_imu [config.yaml]
 *   ./build/calibrate_imu configs/standard3.yaml
 */

#include <fmt/core.h>
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

#include "io/camera.hpp"
#include "io/h30_imu/h30_imu.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

using namespace std::chrono;

const std::string keys =
  "{help h usage ? |                        | Print help                    }"
  "{@config-path   | configs/standard3.yaml | YAML config file path         }";

// ─── Helpers ────────────────────────────────────────────────────────

/// Average a small buffer of quaternions (simple mean + re-normalise).
Eigen::Quaterniond average_quaternions(const std::vector<Eigen::Quaterniond>& qs)
{
  if (qs.empty()) return Eigen::Quaterniond::Identity();

  // Flip quaternions that are in the opposite hemisphere of the first
  Eigen::Vector4d sum = qs[0].coeffs();  // (x, y, z, w) in Eigen ordering
  for (size_t i = 1; i < qs.size(); ++i) {
    Eigen::Vector4d c = qs[i].coeffs();
    if (c.dot(sum) < 0) c = -c;  // flip sign for consistent averaging
    sum += c;
  }
  Eigen::Quaterniond avg;
  avg.coeffs() = sum.normalized();
  return avg;
}

/// Pretty-print a 3×3 matrix with a label.
void print_matrix(const std::string& label, const Eigen::Matrix3d& M)
{
  std::cout << label << ":\n";
  for (int r = 0; r < 3; ++r) {
    std::cout << "  [";
    for (int c = 0; c < 3; ++c) {
      std::cout << std::setw(10) << std::fixed << std::setprecision(6) << M(r, c);
      if (c < 2) std::cout << ", ";
    }
    std::cout << "]\n";
  }
}

/// Format a 3×3 matrix as a YAML-friendly row-major array string.
std::string matrix_to_yaml_string(const Eigen::Matrix3d& M)
{
  std::ostringstream oss;
  oss << "[";
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) {
      oss << std::setprecision(8) << M(r, c);
      if (r != 2 || c != 2) oss << ", ";
    }
  oss << "]";
  return oss.str();
}

/// Draw a large instruction banner on the image.
void draw_instruction(cv::Mat& img, const std::string& line1,
                      const std::string& line2 = "", const cv::Scalar& colour = {0, 255, 255})
{
  int baseline = 0;
  double scale = 0.9;
  int thickness = 2;

  auto sz1 = cv::getTextSize(line1, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
  cv::putText(img, line1, cv::Point((img.cols - sz1.width) / 2, 50),
              cv::FONT_HERSHEY_SIMPLEX, scale, colour, thickness);

  if (!line2.empty()) {
    auto sz2 = cv::getTextSize(line2, cv::FONT_HERSHEY_SIMPLEX, scale * 0.7, thickness, &baseline);
    cv::putText(img, line2, cv::Point((img.cols - sz2.width) / 2, 90),
                cv::FONT_HERSHEY_SIMPLEX, scale * 0.7, {200, 200, 200}, thickness);
  }
}

/// Draw the detected armor plate on the image.
void draw_armor_overlay(cv::Mat& img, const auto_aim::Armor& armor)
{
  if (armor.points.size() >= 4) {
    for (int i = 0; i < 4; ++i)
      cv::line(img, armor.points[i], armor.points[(i + 1) % 4], {0, 255, 0}, 2);
    for (const auto& pt : armor.points) cv::circle(img, pt, 4, {0, 255, 0}, -1);
  }
}

/// Display real-time IMU euler angles and quaternion on the image.
void draw_imu_info(cv::Mat& img, const Eigen::Quaterniond& q, const Eigen::Matrix3d& R_g2ib)
{
  // Compute gimbal YPR via the similarity transform
  Eigen::Matrix3d R_gimbal2world = R_g2ib.transpose() * q.toRotationMatrix() * R_g2ib;
  Eigen::Vector3d ypr = tools::eulers(R_gimbal2world, 2, 1, 0) * 180.0 / M_PI;

  int y0 = img.rows - 130;
  cv::Scalar c{200, 200, 100};
  tools::draw_text(img, fmt::format("IMU q: w={:.3f} x={:.3f} y={:.3f} z={:.3f}",
                                    q.w(), q.x(), q.y(), q.z()),
                   {10, y0}, c, 0.55, 1);
  tools::draw_text(img, fmt::format("Gimbal YPR: {:.1f}  {:.1f}  {:.1f} deg",
                                    ypr[0], ypr[1], ypr[2]),
                   {10, y0 + 30}, c, 0.55, 1);
}

// ─── Main ───────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  // ── Initialise hardware ──
  io::Camera camera(config_path);
  H30IMU imu("/dev/ttyACM1", 460800);
  if (!imu.start()) {
    std::cerr << "[ERROR] Cannot start H30 IMU on /dev/ttyACM1\n";
    return 1;
  }

  auto_aim::YOLO detector(config_path, false);
  auto_aim::Solver solver(config_path);

  // Current R_gimbal2imubody from config (used for live preview)
  Eigen::Matrix3d R_g2ib_current = Eigen::Matrix3d::Identity();
  {
    auto yaml = YAML::LoadFile(config_path);
    auto data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
    R_g2ib_current = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(data.data());
  }

  tools::Exiter exiter;

  cv::namedWindow("IMU Calibration", cv::WINDOW_NORMAL);
  cv::resizeWindow("IMU Calibration", 1280, 720);

  // ── Calibration state machine ──
  enum class State {
    PREVIEW,       // Free look – user sees camera + IMU data
    WAIT_REF,      // Waiting for user to capture reference pose
    COLLECT_REF,   // Collecting samples at reference pose
    WAIT_YAW,      // Waiting for user to yaw right and capture
    COLLECT_YAW,   // Collecting samples at yaw pose
    WAIT_PITCH,    // Waiting for user to pitch up and capture
    COLLECT_PITCH, // Collecting samples at pitch pose
    RESULT,        // Showing result
  };

  State state = State::PREVIEW;

  // Sample buffers
  const int SAMPLES = 60;  // ~1 second at 60Hz camera
  std::vector<Eigen::Quaterniond> sample_buf;
  Eigen::Quaterniond q_ref, q_yaw, q_pitch;

  cv::Mat img;
  std::chrono::steady_clock::time_point t;

  std::cout << "\n"
            << "╔══════════════════════════════════════════════════════════╗\n"
            << "║          IMU-to-Gimbal Calibration Tool                 ║\n"
            << "╠══════════════════════════════════════════════════════════╣\n"
            << "║  Place an armor plate ~1-2m in front of the camera.     ║\n"
            << "║  The armor is used as visual reference only.            ║\n"
            << "║                                                        ║\n"
            << "║  Controls:                                              ║\n"
            << "║    SPACE  – Advance to next calibration step            ║\n"
            << "║    R      – Restart calibration                         ║\n"
            << "║    S      – Save result to YAML config                  ║\n"
            << "║    ESC    – Quit                                        ║\n"
            << "╚══════════════════════════════════════════════════════════╝\n\n";

  Eigen::Matrix3d R_g2ib_result = Eigen::Matrix3d::Identity();
  bool have_result = false;

  while (!exiter.exit()) {
    camera.read(img, t);
    Eigen::Quaterniond q_now = imu.getQuaternionAt(t);

    // Detect armors for visual feedback
    auto armors = detector.detect(img);

    // Draw all detected armors
    for (const auto& armor : armors) draw_armor_overlay(img, armor);

    // Draw IMU info using current (or calibrated) R_gimbal2imubody
    const Eigen::Matrix3d& R_preview = have_result ? R_g2ib_result : R_g2ib_current;
    draw_imu_info(img, q_now, R_preview);

    // Draw crosshair at image centre
    cv::Point ctr(img.cols / 2, img.rows / 2);
    cv::line(img, {ctr.x - 30, ctr.y}, {ctr.x + 30, ctr.y}, {0, 255, 0}, 1);
    cv::line(img, {ctr.x, ctr.y - 30}, {ctr.x, ctr.y + 30}, {0, 255, 0}, 1);

    // Draw number of detected armors
    tools::draw_text(img, fmt::format("Armors: {}", armors.size()),
                     {img.cols - 200, 30}, {0, 255, 0}, 0.6, 1);

    // ── State machine ──
    switch (state) {
      case State::PREVIEW:
        draw_instruction(img,
                         "PREVIEW - Place armor plate in front of camera",
                         "Press SPACE when ready to start calibration");
        break;

      case State::WAIT_REF:
        draw_instruction(img,
                         "STEP 1: Point camera at armor plate, keep gimbal LEVEL and STILL",
                         "Press SPACE to capture reference pose");
        break;

      case State::COLLECT_REF:
        sample_buf.push_back(q_now);
        draw_instruction(img,
                         fmt::format("Collecting reference... {}/{}", sample_buf.size(), SAMPLES),
                         "Keep STILL!");
        if ((int)sample_buf.size() >= SAMPLES) {
          q_ref = average_quaternions(sample_buf);
          sample_buf.clear();
          state = State::WAIT_YAW;
          std::cout << "[Cal] Reference captured: q = ("
                    << q_ref.w() << ", " << q_ref.x() << ", "
                    << q_ref.y() << ", " << q_ref.z() << ")\n";
        }
        break;

      case State::WAIT_YAW:
        draw_instruction(img,
                         "STEP 2: YAW the gimbal ~30 deg to the RIGHT",
                         "Then press SPACE to capture (keep still while capturing)");
        break;

      case State::COLLECT_YAW:
        sample_buf.push_back(q_now);
        draw_instruction(img,
                         fmt::format("Collecting yaw pose... {}/{}", sample_buf.size(), SAMPLES),
                         "Keep STILL!");
        if ((int)sample_buf.size() >= SAMPLES) {
          q_yaw = average_quaternions(sample_buf);
          sample_buf.clear();
          state = State::WAIT_PITCH;
          std::cout << "[Cal] Yaw pose captured: q = ("
                    << q_yaw.w() << ", " << q_yaw.x() << ", "
                    << q_yaw.y() << ", " << q_yaw.z() << ")\n";
        }
        break;

      case State::WAIT_PITCH: {
        draw_instruction(img,
                         "STEP 3: Return to reference, then PITCH the gimbal ~20 deg UP",
                         "Then press SPACE to capture (keep still while capturing)");
        break;
      }

      case State::COLLECT_PITCH:
        sample_buf.push_back(q_now);
        draw_instruction(img,
                         fmt::format("Collecting pitch pose... {}/{}", sample_buf.size(), SAMPLES),
                         "Keep STILL!");
        if ((int)sample_buf.size() >= SAMPLES) {
          q_pitch = average_quaternions(sample_buf);
          sample_buf.clear();
          std::cout << "[Cal] Pitch pose captured: q = ("
                    << q_pitch.w() << ", " << q_pitch.x() << ", "
                    << q_pitch.y() << ", " << q_pitch.z() << ")\n";

          // ── Compute R_gimbal2imubody ──
          // Delta quaternion in IMU body frame: dq_body = q_ref^-1 * q_new
          //   This represents the same physical rotation expressed in IMU body coords.
          //
          // Gimbal FLU convention:
          //   Yaw RIGHT  → rotation vector  (0, 0, -θ)  (negative Z)
          //   Pitch UP   → rotation vector  (0, +φ, 0)  (positive Y)
          //
          // R_gimbal2imubody maps gimbal axes to IMU body axes:
          //   rv_imu = R_gimbal2imubody * rv_gimbal
          //
          // Therefore:
          //   col3 = -normalise(rv_yaw_imu)   (third column, gimbal Z)
          //   col2 =  normalise(rv_pitch_imu) (second column, gimbal Y)
          //   col1 =  col2 × col3             (first column, gimbal X, by right-hand rule)

          // Yaw delta in body frame
          Eigen::Quaterniond dq_yaw = q_ref.conjugate() * q_yaw;
          Eigen::AngleAxisd aa_yaw(dq_yaw);
          Eigen::Vector3d rv_yaw = aa_yaw.axis() * aa_yaw.angle();

          // Pitch delta in body frame
          Eigen::Quaterniond dq_pitch = q_ref.conjugate() * q_pitch;
          Eigen::AngleAxisd aa_pitch(dq_pitch);
          Eigen::Vector3d rv_pitch = aa_pitch.axis() * aa_pitch.angle();

          double yaw_angle_deg = aa_yaw.angle() * 180.0 / M_PI;
          double pitch_angle_deg = aa_pitch.angle() * 180.0 / M_PI;

          std::cout << "\n[Cal] Yaw rotation:  " << yaw_angle_deg << " deg  axis_imu = ["
                    << rv_yaw.normalized().transpose() << "]\n";
          std::cout << "[Cal] Pitch rotation: " << pitch_angle_deg << " deg  axis_imu = ["
                    << rv_pitch.normalized().transpose() << "]\n";

          // Sanity check: motions large enough?
          if (yaw_angle_deg < 10.0) {
            std::cerr << "[Cal] WARNING: Yaw rotation too small (" << yaw_angle_deg
                      << " deg). Aim for ~30 deg. Press R to retry.\n";
          }
          if (pitch_angle_deg < 8.0) {
            std::cerr << "[Cal] WARNING: Pitch rotation too small (" << pitch_angle_deg
                      << " deg). Aim for ~20 deg. Press R to retry.\n";
          }

          // Build R_gimbal2imubody from the two observed axes
          // Yaw right = negative rotation around gimbal Z-up,
          // so gimbal Z in IMU body frame = -rv_yaw.normalised()
          Eigen::Vector3d col3 = -rv_yaw.normalized();

          // Pitch up = positive rotation around gimbal Y-left,
          // so gimbal Y in IMU body frame = rv_pitch.normalised()
          Eigen::Vector3d col2 = rv_pitch.normalized();

          // Orthogonalise: remove col3 component from col2, re-normalise
          col2 = (col2 - col2.dot(col3) * col3).normalized();

          // gimbal X = Y × Z (right-hand rule)
          Eigen::Vector3d col1 = col2.cross(col3);
          col1.normalize();

          R_g2ib_result.col(0) = col1;
          R_g2ib_result.col(1) = col2;
          R_g2ib_result.col(2) = col3;

          // Verify it's a proper rotation (det ≈ +1)
          double det = R_g2ib_result.determinant();
          if (det < 0) {
            std::cout << "[Cal] Flipping sign (det was " << det << ")\n";
            R_g2ib_result.col(0) = -col1;
            det = R_g2ib_result.determinant();
          }

          // Project onto nearest rotation matrix via SVD
          Eigen::JacobiSVD<Eigen::Matrix3d> svd(
              R_g2ib_result, Eigen::ComputeFullU | Eigen::ComputeFullV);
          R_g2ib_result = svd.matrixU() * svd.matrixV().transpose();
          if (R_g2ib_result.determinant() < 0) {
            Eigen::Matrix3d fix = Eigen::Matrix3d::Identity();
            fix(2, 2) = -1;
            R_g2ib_result = svd.matrixU() * fix * svd.matrixV().transpose();
          }

          have_result = true;
          state = State::RESULT;

          std::cout << "\n══════════════════════════════════════════\n";
          print_matrix("R_gimbal2imubody (calibrated)", R_g2ib_result);
          std::cout << "\nYAML format:\n  R_gimbal2imubody: "
                    << matrix_to_yaml_string(R_g2ib_result) << "\n";
          std::cout << "\ndet(R) = " << R_g2ib_result.determinant() << "\n";

          // Verify: compute gimbal YPR at reference pose (should be near 0,0,0
          // for the calibrated matrix, with arbitrary yaw offset)
          Eigen::Matrix3d R_test =
              R_g2ib_result.transpose() * q_ref.toRotationMatrix() * R_g2ib_result;
          Eigen::Vector3d ypr_ref = tools::eulers(R_test, 2, 1, 0) * 180.0 / M_PI;
          std::cout << "\nVerification – Gimbal YPR at reference pose: "
                    << ypr_ref[0] << "  " << ypr_ref[1] << "  " << ypr_ref[2] << " deg\n";
          std::cout << "  (Yaw can be any value; Pitch and Roll should be near 0)\n";

          // Verify yaw pose
          Eigen::Matrix3d R_yaw_test =
              R_g2ib_result.transpose() * q_yaw.toRotationMatrix() * R_g2ib_result;
          Eigen::Vector3d ypr_yaw = tools::eulers(R_yaw_test, 2, 1, 0) * 180.0 / M_PI;
          double yaw_diff = ypr_yaw[0] - ypr_ref[0];
          // wrap to [-180, 180]
          while (yaw_diff > 180) yaw_diff -= 360;
          while (yaw_diff < -180) yaw_diff += 360;
          std::cout << "\nVerification – Yaw pose delta: " << yaw_diff << " deg"
                    << "  (should be ~ -" << yaw_angle_deg << " deg for rightward yaw)\n";

          // Verify pitch pose
          Eigen::Matrix3d R_pitch_test =
              R_g2ib_result.transpose() * q_pitch.toRotationMatrix() * R_g2ib_result;
          Eigen::Vector3d ypr_pitch = tools::eulers(R_pitch_test, 2, 1, 0) * 180.0 / M_PI;
          double pitch_diff = ypr_pitch[1] - ypr_ref[1];
          std::cout << "Verification – Pitch pose delta: " << pitch_diff << " deg"
                    << "  (should be ~ +" << pitch_angle_deg << " deg for upward pitch)\n";

          std::cout << "══════════════════════════════════════════\n\n";
          std::cout << "Press S to save to config, R to retry, ESC to quit.\n";
        }
        break;

      case State::RESULT:
        draw_instruction(img,
                         "CALIBRATION COMPLETE - See terminal for results",
                         "Press S to save | R to retry | ESC to quit",
                         {0, 255, 0});

        // Show live preview with calibrated matrix
        {
          Eigen::Matrix3d R_live =
              R_g2ib_result.transpose() * q_now.toRotationMatrix() * R_g2ib_result;
          Eigen::Vector3d ypr_live = tools::eulers(R_live, 2, 1, 0) * 180.0 / M_PI;
          tools::draw_text(img, fmt::format("CALIBRATED YPR: {:.1f}  {:.1f}  {:.1f} deg",
                                            ypr_live[0], ypr_live[1], ypr_live[2]),
                           {10, img.rows - 170}, {0, 255, 0}, 0.65, 2);
        }
        break;
    }

    // ── Keyboard handling ──
    cv::imshow("IMU Calibration", img);
    int key = cv::waitKey(1);

    if (key == 27) {  // ESC
      break;
    }
    else if (key == ' ') {  // SPACE – advance
      switch (state) {
        case State::PREVIEW:
          state = State::WAIT_REF;
          break;
        case State::WAIT_REF:
          sample_buf.clear();
          state = State::COLLECT_REF;
          std::cout << "[Cal] Collecting reference samples...\n";
          break;
        case State::WAIT_YAW:
          sample_buf.clear();
          state = State::COLLECT_YAW;
          std::cout << "[Cal] Collecting yaw samples...\n";
          break;
        case State::WAIT_PITCH:
          sample_buf.clear();
          state = State::COLLECT_PITCH;
          std::cout << "[Cal] Collecting pitch samples...\n";
          break;
        default:
          break;
      }
    }
    else if (key == 'r' || key == 'R') {  // Restart
      state = State::PREVIEW;
      sample_buf.clear();
      have_result = false;
      R_g2ib_result = Eigen::Matrix3d::Identity();
      std::cout << "[Cal] Restarted.\n";
    }
    else if (key == 's' || key == 'S') {  // Save
      if (!have_result) {
        std::cout << "[Cal] No calibration result to save. Complete calibration first.\n";
        continue;
      }

      // Read original YAML, replace R_gimbal2imubody line, write back
      std::string yaml_str = matrix_to_yaml_string(R_g2ib_result);

      // Read file
      std::ifstream fin(config_path);
      if (!fin.is_open()) {
        std::cerr << "[Cal] Cannot open " << config_path << " for reading.\n";
        continue;
      }
      std::vector<std::string> lines;
      std::string line;
      while (std::getline(fin, line)) lines.push_back(line);
      fin.close();

      // Find and replace R_gimbal2imubody line
      bool found = false;
      for (auto& l : lines) {
        if (l.find("R_gimbal2imubody:") != std::string::npos &&
            l.find("#") != 0) {  // skip pure comment lines
          l = "R_gimbal2imubody: " + yaml_str;
          found = true;
          break;
        }
      }

      if (!found) {
        std::cerr << "[Cal] Could not find 'R_gimbal2imubody:' in " << config_path << "\n";
        continue;
      }

      // Remove old comment lines about FRD/FLU above R_gimbal2imubody and add new ones
      std::ofstream fout(config_path);
      if (!fout.is_open()) {
        std::cerr << "[Cal] Cannot open " << config_path << " for writing.\n";
        continue;
      }
      for (const auto& l : lines) fout << l << "\n";
      fout.close();

      std::cout << "\n[Cal] ✅ Saved R_gimbal2imubody to " << config_path << "\n";
      std::cout << "     " << yaml_str << "\n\n";
    }
  }

  cv::destroyAllWindows();
  std::cout << "[Cal] Done.\n";
  return 0;
}
