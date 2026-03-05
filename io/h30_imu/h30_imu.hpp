#pragma once

#include <chrono>
#include <deque>
#include <thread>
#include <atomic>
#include <mutex>
#include <Eigen/Geometry>
#include <string>

class H30IMU {
public:
    /**
     * @param device  Serial device, e.g. "/dev/ttyACM0"
     * @param baudrate Baud rate (e.g., 115200, 460800)
     * @param timeout_sec Timeout for read operations (seconds, used for serial VMIN/ VTIME)
     */
    H30IMU(const std::string& device, int baudrate, float timeout_sec = 0.2f);
    ~H30IMU();

    /// Start the background reading thread.
    bool start();

    /// Stop the thread and close the device.
    void stop();

    /**
     * @brief Get interpolated quaternion at a specific time point.
     * @param tp Time point (std::chrono::steady_clock)
     * @return Interpolated quaternion, or identity if buffer empty.
     */
    Eigen::Quaterniond getQuaternionAt(std::chrono::steady_clock::time_point tp) const;

private:
    void workerFunction();
    bool openSerial();
    void closeSerial();
    int setInterfaceAttrs(int fd, int speed, float timeout_sec);

    // Serial port members
    std::string device_;
    int baudrate_;
    float timeout_sec_;
    int fd_;                                 // file descriptor for serial port
    static constexpr size_t READ_BUF_SIZE = 256;

    // Ring buffer for interpolated samples
    struct Sample {
        std::chrono::steady_clock::time_point timestamp;
        Eigen::Quaterniond quat;
    };
    mutable std::mutex mutex_;
    std::deque<Sample> buffer_;
    const size_t max_buffer_size_ = 200;      // adjust as needed

    std::atomic<bool> running_{false};
    std::thread worker_;
};