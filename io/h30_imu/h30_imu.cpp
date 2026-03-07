#include "h30_imu.hpp"
#include <algorithm>      // std::clamp, std::upper_bound
#include <fcntl.h>        // open, O_RDWR, etc.
#include <unistd.h>       // read, write, close
#include <termios.h>      // tcgetattr, tcsetattr, cfsetispeed, etc.
#include <cstring>        // memset
#include <cerrno>
#include <iostream>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---- H30 Protocol Constants ----
static constexpr uint8_t HDR0        = 0x59;
static constexpr uint8_t HDR1        = 0x53;
static constexpr uint8_t MAX_PAYLOAD = 255;

// TLV data IDs
static constexpr uint8_t ID_EULER = 0x40;  // 12 bytes: pitch, roll, yaw (int32 × 1e-6 deg)
static constexpr uint8_t ID_QUAT  = 0x41;  // 16 bytes: w, x, y, z   (int32 × 1e-6)

// ---- Little-endian helpers ----
static inline uint16_t read_u16_le(const uint8_t *p) {
    return static_cast<uint16_t>(p[0]) |
           static_cast<uint16_t>(p[1]) << 8;
}

static inline int32_t read_i32_le(const uint8_t *p) {
    uint32_t v = static_cast<uint32_t>(p[0])
               | (static_cast<uint32_t>(p[1]) << 8)
               | (static_cast<uint32_t>(p[2]) << 16)
               | (static_cast<uint32_t>(p[3]) << 24);
    return static_cast<int32_t>(v);
}

// ---- Fletcher-like checksum ----
static void fletcher_checksum(const std::vector<uint8_t> &data,
                              uint8_t &ck1, uint8_t &ck2) {
    ck1 = 0;
    ck2 = 0;
    for (uint8_t b : data) {
        ck1 = static_cast<uint8_t>((ck1 + b) & 0xFF);
        ck2 = static_cast<uint8_t>((ck2 + ck1) & 0xFF);
    }
}

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
    while (running_) {
        IMUFrame frame;
        if (!readFrame(frame)) {
            continue;
        }

        Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
        bool got_orientation = false;

        // Prefer quaternion if available
        if (frame.has_quat) {
            q = Eigen::Quaterniond(
                static_cast<double>(frame.quat[0]),   // w
                static_cast<double>(frame.quat[1]),   // x
                static_cast<double>(frame.quat[2]),   // y
                static_cast<double>(frame.quat[3])    // z
            ).normalized();
            got_orientation = true;
        }
        // Fallback: convert euler angles to quaternion
        else if (frame.has_euler) {
            double pitch = frame.euler_deg[0] * M_PI / 180.0;
            double roll  = frame.euler_deg[1] * M_PI / 180.0;
            double yaw   = frame.euler_deg[2] * M_PI / 180.0;
            q = Eigen::AngleAxisd(yaw,   Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(roll,  Eigen::Vector3d::UnitX());
            q.normalize();
            got_orientation = true;
        }

        if (got_orientation) {
            Sample sample;
            sample.timestamp = std::chrono::steady_clock::now();
            sample.quat = q;

            std::lock_guard<std::mutex> lock(mutex_);
            buffer_.push_back(sample);
            while (buffer_.size() > max_buffer_size_)
                buffer_.pop_front();
        }
    }
}

bool H30IMU::readExact(uint8_t* buf, size_t n) {
    size_t total = 0;
    while (total < n && running_) {
        ssize_t ret = ::read(fd_, buf + total, n - total);
        if (ret < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }
            return false;
        } else if (ret == 0) {
            return false;
        }
        total += static_cast<size_t>(ret);
    }
    return total == n;
}

bool H30IMU::readFrame(IMUFrame& out) {
    // 1) Sync on header bytes: 0x59 0x53
    uint8_t prev = 0, b = 0;
    bool synced = false;

    while (running_) {
        ssize_t ret = ::read(fd_, &b, 1);
        if (ret == 1) {
            if (prev == HDR0 && b == HDR1) {
                synced = true;
                break;
            }
            prev = b;
        } else if (ret < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }
            return false;
        } else {
            return false;
        }
    }
    if (!synced) return false;

    // 2) Read seq(2) + len(1)
    uint8_t head[3];
    if (!readExact(head, 3)) return false;

    uint16_t seq    = read_u16_le(head);
    uint8_t  length = head[2];

    if (length > MAX_PAYLOAD) {
        // Bad length — skip and resync
        uint8_t dump[256];
        (void)::read(fd_, dump, std::min((size_t)length, sizeof(dump)));
        return false;
    }

    // 3) Read payload
    std::vector<uint8_t> payload(length);
    if (!readExact(payload.data(), length)) return false;

    // 4) Read and verify checksum
    uint8_t ck[2];
    if (!readExact(ck, 2)) return false;

    // Checksum covers: seq(2 bytes LE) + length(1) + payload
    std::vector<uint8_t> chk_data;
    chk_data.reserve(3 + length);
    chk_data.push_back(static_cast<uint8_t>(seq & 0xFF));
    chk_data.push_back(static_cast<uint8_t>((seq >> 8) & 0xFF));
    chk_data.push_back(length);
    chk_data.insert(chk_data.end(), payload.begin(), payload.end());

    uint8_t ck1, ck2;
    fletcher_checksum(chk_data, ck1, ck2);
    if (ck1 != ck[0] || ck2 != ck[1]) {
        return false;  // checksum mismatch, skip frame
    }

    // 5) Parse TLV blocks from payload
    out = IMUFrame{};
    out.seq = seq;

    size_t i = 0;
    while (i + 2 <= payload.size()) {
        uint8_t data_id  = payload[i++];
        uint8_t data_len = payload[i++];
        if (i + data_len > payload.size()) break;

        switch (data_id) {
        case ID_EULER:
            if (data_len == 12) {
                int32_t pitch = read_i32_le(&payload[i + 0]);
                int32_t roll  = read_i32_le(&payload[i + 4]);
                int32_t yaw   = read_i32_le(&payload[i + 8]);
                out.euler_deg[0] = pitch * 1e-6f;
                out.euler_deg[1] = roll  * 1e-6f;
                out.euler_deg[2] = yaw   * 1e-6f;
                out.has_euler = true;
            }
            break;

        case ID_QUAT:
            if (data_len == 16) {
                int32_t q0 = read_i32_le(&payload[i + 0]);
                int32_t q1 = read_i32_le(&payload[i + 4]);
                int32_t q2 = read_i32_le(&payload[i + 8]);
                int32_t q3 = read_i32_le(&payload[i + 12]);
                out.quat[0] = q0 * 1e-6f;  // w
                out.quat[1] = q1 * 1e-6f;  // x
                out.quat[2] = q2 * 1e-6f;  // y
                out.quat[3] = q3 * 1e-6f;  // z
                out.has_quat = true;
            }
            break;

        default:
            // Skip unknown TLV blocks (accel, gyro, mag, temp, etc.)
            break;
        }

        i += data_len;
    }

    return true;
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