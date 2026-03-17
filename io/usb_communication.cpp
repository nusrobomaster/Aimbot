#include "usb_communication.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <sys/select.h>
#include <iostream>

namespace calibur {

/* ────────────────────────────────────────────────────────────────────────── */
/* Construction / Destruction                                                */
/* ────────────────────────────────────────────────────────────────────────── */

USBCommunication::USBCommunication(const std::string& device_path)
    : device_path_(device_path)
    , fd_(-1)
    , is_open_(false) {}

USBCommunication::~USBCommunication() {
    close();
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Open / Close / Configure                                                  */
/* ────────────────────────────────────────────────────────────────────────── */

bool USBCommunication::open() {
    if (is_open_) return true;

    fd_ = ::open(device_path_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
        std::cerr << "[USB] Failed to open " << device_path_
                  << ": " << strerror(errno) << std::endl;
        return false;
    }

    if (!configure()) {
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    // Clear O_NONBLOCK after configure so RX can use select()
    int flags = fcntl(fd_, F_GETFL, 0);
    fcntl(fd_, F_SETFL, flags & ~O_NONBLOCK);

    // Flush stale data
    tcflush(fd_, TCIOFLUSH);

    is_open_ = true;

    // Start RX thread
    rx_running_ = true;
    rx_thread_ = std::thread(&USBCommunication::rxLoop, this);

    std::cout << "[USB] Opened " << device_path_ << " — RX thread started." << std::endl;
    return true;
}

bool USBCommunication::close() {
    // Stop RX thread
    rx_running_ = false;
    if (rx_thread_.joinable()) rx_thread_.join();

    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    is_open_ = false;
    return true;
}

bool USBCommunication::isOpen() const {
    return is_open_;
}

bool USBCommunication::configure(int baudrate) {
    if (fd_ < 0) return false;

    struct termios tty{};
    if (tcgetattr(fd_, &tty) != 0) {
        std::cerr << "[USB] tcgetattr failed: " << strerror(errno) << std::endl;
        return false;
    }

    // Raw mode
    cfmakeraw(&tty);

    // 8N1, no flow control
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag &= ~(PARENB | CSTOPB | CRTSCTS);
    tty.c_cflag |= CLOCAL | CREAD;

    // Baud rate (CDC ignores this, but set for completeness)
    speed_t speed = B115200;
    switch (baudrate) {
        case 9600:    speed = B9600;    break;
        case 19200:   speed = B19200;   break;
        case 38400:   speed = B38400;   break;
        case 57600:   speed = B57600;   break;
        case 115200:  speed = B115200;  break;
        case 230400:  speed = B230400;  break;
        case 460800:  speed = B460800;  break;
        case 921600:  speed = B921600;  break;
        default:      speed = B115200;  break;
    }
    cfsetispeed(&tty, speed);
    cfsetospeed(&tty, speed);

    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 1;  // 100ms timeout

    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
        std::cerr << "[USB] tcsetattr failed: " << strerror(errno) << std::endl;
        return false;
    }

    return true;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* TX — Raw send: [0x7F] [ID] [payload bytes]                               */
/* ────────────────────────────────────────────────────────────────────────── */

void USBCommunication::sendRaw(uint8_t packet_id, const void* data, uint16_t size) {
    if (fd_ < 0 || !is_open_ || size > 250) return;

    uint8_t buf[256];
    buf[0] = Protocol::MAGIC_BYTE;  // 0x7F
    buf[1] = packet_id;
    std::memcpy(&buf[2], data, size);

    std::lock_guard<std::mutex> lock(tx_mu_);
    ssize_t written = ::write(fd_, buf, 2 + size);
    if (written < 0) {
        std::cerr << "[USB] write failed: " << strerror(errno) << std::endl;
    }
    tx_pkt_count_++;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* TX — Typed command senders                                                */
/* ────────────────────────────────────────────────────────────────────────── */

void USBCommunication::sendGimbalCommand(float yaw, float pitch) {
    GimbalCommandPacket pkt{};
    pkt.yaw   = yaw;
    pkt.pitch = pitch;
    make_reliable(pkt);
    sendRaw(Protocol::ID_GIMBAL_COMMAND, &pkt, sizeof(pkt));
}

void USBCommunication::sendFiringCommand(bool fire) {
    FiringCommandPacket pkt{};
    pkt.fire_state = fire;
    make_reliable(pkt);
    sendRaw(Protocol::ID_FIRING_COMMAND, &pkt, sizeof(pkt));
}

void USBCommunication::sendChassisSpeed(float vx, float vy, float vyaw) {
    ChassisSpeedCommandPacket pkt{};
    pkt.V_horz = vx;
    pkt.V_lat  = vy;
    pkt.V_yaw  = vyaw;
    make_reliable(pkt);
    sendRaw(Protocol::ID_CHASSIS_SPEED, &pkt, sizeof(pkt));
}

void USBCommunication::sendSurveilCommand(bool state) {
    SurveilCommandPacket pkt{};
    pkt.surveillance_state = state;
    make_reliable(pkt);
    sendRaw(Protocol::ID_SURVEIL_COMMAND, &pkt, sizeof(pkt));
}

void USBCommunication::sendAimCommand(bool state) {
    AimCommandPacket pkt{};
    pkt.aiming_state = state;
    make_reliable(pkt);
    sendRaw(Protocol::ID_AIM_COMMAND, &pkt, sizeof(pkt));
}

void USBCommunication::sendNavigatingCommand(bool state) {
    IsNavigatingPacket pkt{};
    pkt.navigating_state = state;
    make_reliable(pkt);
    sendRaw(Protocol::ID_IS_NAVIGATING, &pkt, sizeof(pkt));
}

/* ────────────────────────────────────────────────────────────────────────── */
/* RX — Status snapshot                                                      */
/* ────────────────────────────────────────────────────────────────────────── */

BoardStatus USBCommunication::getStatus() const {
    std::lock_guard<std::mutex> lock(status_mu_);
    return status_;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* RX — Payload length lookup (mirrors firmware switch table)                */
/* ────────────────────────────────────────────────────────────────────────── */

uint16_t USBCommunication::payloadLenForId(uint8_t id) const {
    switch (id) {
        case Protocol::ID_GIMBAL_JOINTS:      return sizeof(GimbalJointsPacket);
        case Protocol::ID_COMPETITION_STATUS: return sizeof(CompetitionStatusPacket);
        case Protocol::ID_GIMBAL_COMMAND:     return sizeof(GimbalCommandPacket);
        case Protocol::ID_FIRING_COMMAND:     return sizeof(FiringCommandPacket);
        case Protocol::ID_CHASSIS_SPEED:      return sizeof(ChassisSpeedCommandPacket);
        case Protocol::ID_DUMMY:              return sizeof(DummyPacket);
        case Protocol::ID_SURVEIL_COMMAND:    return sizeof(SurveilCommandPacket);
        case Protocol::ID_AIM_COMMAND:        return sizeof(AimCommandPacket);
        case Protocol::ID_IS_NAVIGATING:      return sizeof(IsNavigatingPacket);
        default:                              return 0;
    }
}

/* ────────────────────────────────────────────────────────────────────────── */
/* RX — Background thread: read + state-machine parser                       */
/* ────────────────────────────────────────────────────────────────────────── */

void USBCommunication::rxLoop() {
    enum class State : uint8_t { WAIT_MAGIC, WAIT_ID, WAIT_DATA };

    State    state       = State::WAIT_MAGIC;
    uint8_t  pkt_id      = 0;
    uint16_t payload_len = 0;
    uint16_t payload_pos = 0;
    uint8_t  payload_buf[Protocol::MAX_PAYLOAD_SIZE];
    uint8_t  read_buf[512];

    while (rx_running_) {
        // select() with 50ms timeout so we can check rx_running_
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(fd_, &rfds);
        struct timeval tv;
        tv.tv_sec  = 0;
        tv.tv_usec = 50000;

        int sel = select(fd_ + 1, &rfds, nullptr, nullptr, &tv);
        if (sel <= 0) continue;

        ssize_t n = ::read(fd_, read_buf, sizeof(read_buf));
        if (n <= 0) continue;

        for (ssize_t i = 0; i < n; ++i) {
            uint8_t byte = read_buf[i];

            switch (state) {
                case State::WAIT_MAGIC:
                    if (byte == Protocol::MAGIC_BYTE) {
                        state = State::WAIT_ID;
                    }
                    break;

                case State::WAIT_ID:
                    pkt_id = byte;
                    payload_len = payloadLenForId(pkt_id);
                    if (payload_len > 0 && payload_len <= Protocol::MAX_PAYLOAD_SIZE) {
                        payload_pos = 0;
                        state = State::WAIT_DATA;
                    } else {
                        state = State::WAIT_MAGIC;  // unknown ID, resync
                    }
                    break;

                case State::WAIT_DATA:
                    payload_buf[payload_pos++] = byte;
                    if (payload_pos >= payload_len) {
                        handleRxPacket(pkt_id, payload_buf, payload_len);
                        state = State::WAIT_MAGIC;
                    }
                    break;
            }
        }
    }
}

/* ────────────────────────────────────────────────────────────────────────── */
/* RX — Packet dispatch                                                      */
/* ────────────────────────────────────────────────────────────────────────── */

void USBCommunication::handleRxPacket(uint8_t id, const uint8_t* payload, uint16_t len) {
    switch (id) {
        case Protocol::ID_GIMBAL_JOINTS: {
            if (len != sizeof(GimbalJointsPacket)) break;
            auto* p = reinterpret_cast<const GimbalJointsPacket*>(payload);
            if (!is_reliable(*p)) { rx_crc_fail_++; break; }
            std::lock_guard<std::mutex> lock(status_mu_);
            status_.yaw_angle   = p->yaw_angle;
            status_.pitch_angle = p->pitch_angle;
            break;
        }

        case Protocol::ID_COMPETITION_STATUS: {
            if (len != sizeof(CompetitionStatusPacket)) break;
            auto* p = reinterpret_cast<const CompetitionStatusPacket*>(payload);
            if (!is_reliable(*p)) { rx_crc_fail_++; break; }
            std::lock_guard<std::mutex> lock(status_mu_);
            status_.game_progress    = p->game_progress;
            status_.time_left        = p->time_left;
            status_.robot_id         = p->robot_id;
            status_.current_hp       = p->current_hp;
            status_.red_hero_hp      = p->red_hero_hp;
            status_.red_standard_hp  = p->red_standard_hp;
            status_.red_sentry_hp    = p->red_sentry_hp;
            status_.blue_hero_hp     = p->blue_hero_hp;
            status_.blue_standard_hp = p->blue_standard_hp;
            status_.blue_sentry_hp   = p->blue_sentry_hp;
            break;
        }

        default:
            break;
    }
    rx_pkt_count_++;
}

} // namespace calibur