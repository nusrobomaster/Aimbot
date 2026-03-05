#include "h30_imu.hpp"
#include <algorithm>      // std::clamp, std::upper_bound
#include <fcntl.h>        // open, O_RDWR, etc.
#include <unistd.h>       // read, write, close
#include <termios.h>      // tcgetattr, tcsetattr, cfsetispeed, etc.
#include <cstring>        // memset
#include <iostream>       // for error logging (replace with your logger)

// Utility to convert baud rate constant (e.g., B115200) from integer speed.
// You may need to expand this table for other baud rates.
static speed_t getBaudRate(int baud) {
    switch (baud) {
        case 9600:   return B9600;
        case 19200:  return B19200;
        case 38400:  return B38400;
        case 57600:  return B57600;
        case 115200: return B115200;
        case 230400: return B230400;
        case 460800: return B460800;
        case 500000: return B500000;
        case 576000: return B576000;
        case 921600: return B921600;
        default:     return B115200;   // fallback
    }
}

H30IMU::H30IMU(const std::string& device, int baudrate, float timeout_sec)
    : device_(device), baudrate_(baudrate), timeout_sec_(timeout_sec), fd_(-1) {}

H30IMU::~H30IMU() {
    stop();
}

bool H30IMU::start() {
    if (running_) return false;
    if (!openSerial()) return false;
    running_ = true;
    worker_ = std::thread(&H30IMU::workerFunction, this);
    return true;
}

void H30IMU::stop() {
    running_ = false;
    if (worker_.joinable())
        worker_.join();
    closeSerial();
}

bool H30IMU::openSerial() {
    fd_ = open(device_.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    if (fd_ < 0) {
        std::cerr << "[H30IMU] Failed to open " << device_ << std::endl;
        return false;
    }
    if (setInterfaceAttrs(fd_, baudrate_, timeout_sec_) != 0) {
        close(fd_);
        fd_ = -1;
        return false;
    }
    return true;
}

void H30IMU::closeSerial() {
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

int H30IMU::setInterfaceAttrs(int fd, int speed, float timeout_sec) {
    struct termios tty;
    memset(&tty, 0, sizeof(tty));
    if (tcgetattr(fd, &tty) != 0) {
        std::cerr << "[H30IMU] tcgetattr error" << std::endl;
        return -1;
    }

    cfsetospeed(&tty, getBaudRate(speed));
    cfsetispeed(&tty, getBaudRate(speed));

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;     // 8-bit chars
    tty.c_iflag &= ~IGNBRK;                         // disable break processing
    tty.c_lflag = 0;                                 // no signaling, echo, etc.
    tty.c_oflag = 0;
    tty.c_cc[VMIN]  = 0;                             // read doesn't block
    // VMIN=0, VTIME in deciseconds: timeout = timeout_sec * 10
    tty.c_cc[VTIME] = static_cast<cc_t>(timeout_sec * 10);

    tty.c_iflag &= ~(IXON | IXOFF | IXANY);          // shut off flow control
    tty.c_cflag |= (CLOCAL | CREAD);                  // ignore modem controls
    tty.c_cflag &= ~(PARENB | PARODD);                // no parity
    tty.c_cflag &= ~CSTOPB;                           // 1 stop bit
    tty.c_cflag &= ~CRTSCTS;                          // no hardware flow control

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        std::cerr << "[H30IMU] tcsetattr error" << std::endl;
        return -1;
    }
    return 0;
}

void H30IMU::workerFunction() {
    // Buffer for raw bytes
    uint8_t raw_buf[READ_BUF_SIZE];

    // --------------------------------------------------------------------
    // TODO: Define your IMU's binary protocol here.
    // You need to implement a state machine that reads bytes,
    // finds packet boundaries, and extracts:
    //   - quaternion (4 floats, probably in [w, x, y, z] or [x, y, z, w])
    //   - timestamp (optional, microseconds since some epoch)
    // Then push a Sample with that timestamp and quaternion.
    //
    // Below is a **placeholder** that assumes each read returns a complete
    // packet of 16 bytes (4 floats). You must replace it with real parsing.
    // --------------------------------------------------------------------

    while (running_) {
        ssize_t n = read(fd_, raw_buf, sizeof(raw_buf));
        if (n <= 0) {
            // No data or error – sleep a bit and retry
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // --- Placeholder parsing (replace with your actual protocol) ---
        // Assume each packet is exactly 16 bytes: 4 floats, little‑endian,
        // in order [w, x, y, z]. Also assume the packet starts at raw_buf[0].
        if (n >= 16) {
            // Convert bytes to floats (assuming little‑endian)
            float* pf = reinterpret_cast<float*>(raw_buf);
            float w = pf[0];
            float x = pf[1];
            float y = pf[2];
            float z = pf[3];

            // Optional: timestamp could be another 8 bytes (uint64_t)
            // uint64_t ts_us = *reinterpret_cast<uint64_t*>(raw_buf + 16);

            Sample sample;
            sample.timestamp = std::chrono::steady_clock::now();  // use system time
            sample.quat = Eigen::Quaterniond(w, x, y, z).normalized();

            // Add to ring buffer
            {
                std::lock_guard<std::mutex> lock(mutex_);
                buffer_.push_back(sample);
                while (buffer_.size() > max_buffer_size_)
                    buffer_.pop_front();
            }
        }
        // -----------------------------------------------------------------
    }
}

Eigen::Quaterniond H30IMU::getQuaternionAt(std::chrono::steady_clock::time_point tp) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (buffer_.empty())
        return Eigen::Quaterniond::Identity();

    // Find first sample with timestamp > tp
    auto it = std::upper_bound(buffer_.begin(), buffer_.end(), tp,
        [](const auto& t, const Sample& s) { return t < s.timestamp; });

    if (it == buffer_.begin())
        return buffer_.front().quat;          // tp older than oldest sample
    if (it == buffer_.end())
        return buffer_.back().quat;            // tp newer than newest sample

    auto after = it;
    auto before = std::prev(it);

    using dur_d = std::chrono::duration<double>;
    double k = dur_d(tp - before->timestamp).count() /
               dur_d(after->timestamp - before->timestamp).count();
    k = std::clamp(k, 0.0, 1.0);

    return before->quat.slerp(k, after->quat).normalized();
}