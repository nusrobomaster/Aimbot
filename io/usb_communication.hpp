#ifndef __CALIBUR_USB_COMMUNICATION_H__
#define __CALIBUR_USB_COMMUNICATION_H__

#include <string>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <thread>
#include <atomic>
#include <functional>

namespace calibur {

/* ────────────────────────────────────────────────────────────────────────── */
/* Protocol constants (must match usb_task.h on STM32)                      */
/* ────────────────────────────────────────────────────────────────────────── */
namespace Protocol {
    constexpr uint8_t  MAGIC_BYTE       = 0x7F;
    constexpr uint16_t MAX_PAYLOAD_SIZE = 256;

    // Packet IDs
    constexpr uint8_t ID_GIMBAL_JOINTS      = 3;
    constexpr uint8_t ID_DUMMY              = 4;
    constexpr uint8_t ID_CHASSIS_SPEED      = 6;
    constexpr uint8_t ID_COMPETITION_STATUS = 11;
    constexpr uint8_t ID_GIMBAL_COMMAND     = 12;
    constexpr uint8_t ID_FIRING_COMMAND     = 13;
    constexpr uint8_t ID_CV_DETECTED        = 14;
    constexpr uint8_t ID_SURVEIL_COMMAND    = 15;
    constexpr uint8_t ID_AIM_COMMAND        = 16;
    constexpr uint8_t ID_IS_NAVIGATING      = 17;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* CRC-16 (identical to STM32 firmware usb_task.h)                          */
/* ────────────────────────────────────────────────────────────────────────── */
inline uint16_t usb_crc16(const uint8_t* data, uint16_t size) {
    uint16_t crc = 0xFFFF;
    while (size--) {
        uint8_t x = (crc >> 8) ^ *data++;
        x ^= x >> 4;
        crc = (crc << 8)
            ^ (static_cast<uint16_t>(x) << 12)
            ^ (static_cast<uint16_t>(x) << 5)
            ^ (static_cast<uint16_t>(x));
    }
    return crc;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Packet structs (packed, trailing CRC — mirrors firmware RELIABLE_PACKET)  */
/* ────────────────────────────────────────────────────────────────────────── */
#pragma pack(push, 1)

// ── TX: Aimbot → STM32 ─────────────────────────────────────────────────

struct GimbalCommandPacket {
    float    yaw;
    float    pitch;
    uint16_t crc;
};

struct FiringCommandPacket {
    bool     fire_state;
    uint16_t crc;
};

struct ChassisSpeedCommandPacket {
    float    V_horz;
    float    V_lat;
    float    V_yaw;
    uint16_t crc;
};

struct SurveilCommandPacket {
    bool     surveillance_state;
    uint16_t crc;
};

struct AimCommandPacket {
    bool     aiming_state;
    uint16_t crc;
};

struct IsNavigatingPacket {
    bool     navigating_state;
    uint16_t crc;
};

// ── RX: STM32 → Aimbot ─────────────────────────────────────────────────

struct GimbalJointsPacket {
    float    yaw_angle;
    float    pitch_angle;
    uint16_t crc;
};

struct CompetitionStatusPacket {
    uint16_t game_progress;
    uint16_t time_left;
    uint16_t robot_id;
    uint16_t current_hp;
    uint16_t red_hero_hp;
    uint16_t red_standard_hp;
    uint16_t red_sentry_hp;
    uint16_t blue_hero_hp;
    uint16_t blue_standard_hp;
    uint16_t blue_sentry_hp;
    uint16_t crc;
};

struct DummyPacket {
    int      num1;
    int      num2;
    int      num3;
    uint16_t id;
    uint16_t crc;
};

#pragma pack(pop)

/* ────────────────────────────────────────────────────────────────────────── */
/* CRC helpers                                                               */
/* ────────────────────────────────────────────────────────────────────────── */
template <typename T>
inline void make_reliable(T& pkt) {
    pkt.crc = usb_crc16(reinterpret_cast<const uint8_t*>(&pkt), sizeof(T) - 2);
}

template <typename T>
inline bool is_reliable(const T& pkt) {
    return pkt.crc == usb_crc16(reinterpret_cast<const uint8_t*>(&pkt), sizeof(T) - 2);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Board status snapshot (populated by RX thread)                            */
/* ────────────────────────────────────────────────────────────────────────── */
struct BoardStatus {
    // From GimbalJointsPacket (50 Hz)
    float yaw_angle   = 0.0f;
    float pitch_angle = 0.0f;

    // From CompetitionStatusPacket (10 Hz)
    uint16_t game_progress    = 0;
    uint16_t time_left        = 0;
    uint16_t robot_id         = 0;
    uint16_t current_hp       = 0;
    uint16_t red_hero_hp      = 0;
    uint16_t red_standard_hp  = 0;
    uint16_t red_sentry_hp    = 0;
    uint16_t blue_hero_hp     = 0;
    uint16_t blue_standard_hp = 0;
    uint16_t blue_sentry_hp   = 0;
};

/* ────────────────────────────────────────────────────────────────────────── */
/* USBCommunication                                                          */
/* ────────────────────────────────────────────────────────────────────────── */
class USBCommunication {
public:
    explicit USBCommunication(const std::string& device_path);
    ~USBCommunication();

    USBCommunication(const USBCommunication&) = delete;
    USBCommunication& operator=(const USBCommunication&) = delete;

    bool open();
    bool close();
    bool isOpen() const;
    bool configure(int baudrate = 115200);

    // ── TX commands (thread-safe) ──────────────────────────────────────
    void sendGimbalCommand(float yaw, float pitch);
    void sendFiringCommand(bool fire);
    void sendChassisSpeed(float vx, float vy, float vyaw);
    void sendSurveilCommand(bool state);
    void sendAimCommand(bool state);
    void sendNavigatingCommand(bool state);

    // ── RX status ──────────────────────────────────────────────────────
    BoardStatus getStatus() const;

    // ── Diagnostics ────────────────────────────────────────────────────
    uint32_t rxPacketCount() const  { return rx_pkt_count_.load(); }
    uint32_t rxCrcFailCount() const { return rx_crc_fail_.load(); }
    uint32_t txPacketCount() const  { return tx_pkt_count_.load(); }

private:
    // Serial port
    std::string device_path_;
    int fd_;
    bool is_open_;

    // TX
    std::mutex tx_mu_;
    void sendRaw(uint8_t packet_id, const void* data, uint16_t size);

    // RX thread
    std::thread rx_thread_;
    std::atomic<bool> rx_running_{false};
    void rxLoop();
    void handleRxPacket(uint8_t id, const uint8_t* payload, uint16_t len);
    uint16_t payloadLenForId(uint8_t id) const;

    // RX state
    mutable std::mutex status_mu_;
    BoardStatus status_;

    // Stats
    std::atomic<uint32_t> rx_pkt_count_{0};
    std::atomic<uint32_t> rx_crc_fail_{0};
    std::atomic<uint32_t> tx_pkt_count_{0};
};

} // namespace calibur

#endif // __CALIBUR_USB_COMMUNICATION_H__