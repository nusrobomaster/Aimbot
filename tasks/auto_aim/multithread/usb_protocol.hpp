// tasks/auto_aim/multithread/usb_protocol.hpp
#pragma once

#include <cstdint>
#include <vector>
#include <cstring>

namespace protocol {

// Match MCU defines from usb_task.c
constexpr uint8_t USB_MAGIC_BYTE = 0x7F;  // Preamble byte
constexpr size_t USB_MAX_PAYLOAD_SIZE = 250;

// Protocol IDs for NUS25 (Matching MCU)
constexpr uint8_t ID_DUMMY = 4;
constexpr uint8_t ID_CHASSIS_SPEED = 6;
constexpr uint8_t ID_LEFT_TRIGGER = 8;
constexpr uint8_t ID_CHASSIS_SPIN = 10;
constexpr uint8_t ID_COMPETITION_STATUS = 11;
constexpr uint8_t ID_GIMBAL_COMMAND = 12;     // <-- Use this
constexpr uint8_t ID_FIRING_COMMAND = 13;      // <-- Use this
constexpr uint8_t ID_CV_DETECTED = 14;
constexpr uint8_t ID_SURVEIL_COMMAND = 15;
constexpr uint8_t ID_AIM_COMMAND = 16;
constexpr uint8_t ID_IS_NAVIGATING = 17;
constexpr uint8_t ID_OCCUPATION_STATUS = 19;
constexpr uint8_t ID_WIN_STATUS = 20;

// Reliable packet marker (matches MCU's MAKE_RELIABLE macro)
constexpr uint32_t RELIABLE_MARKER = 0x12345678;

// Packet structures - EXACTLY matching MCU definitions
struct __attribute__((packed)) cvGimbalCommandPacket {
    float yaw;
    float pitch;
    uint32_t __reliable;  // For MAKE_RELIABLE macro
};

struct __attribute__((packed)) firingCommandPacket {
    uint8_t fire_state;   // 0: Stop firing, 1: Start firing
    uint32_t __reliable;  // For MAKE_RELIABLE macro
};

struct __attribute__((packed)) chassisSpeedCommandPacket {
    float V_horz;   // vx
    float V_lat;    // vy  
    float V_yaw;    // vz
    uint32_t __reliable;
};

// Simple send functions - just add preamble + ID + payload
inline std::vector<uint8_t> packGimbalCommand(float yaw, float pitch) {
    std::vector<uint8_t> buffer;
    buffer.reserve(2 + sizeof(cvGimbalCommandPacket));
    
    // Format: [MAGIC][ID][PAYLOAD]
    buffer.push_back(USB_MAGIC_BYTE);
    buffer.push_back(ID_GIMBAL_COMMAND);
    
    cvGimbalCommandPacket pkt;
    pkt.yaw = yaw;
    pkt.pitch = pitch;
    pkt.__reliable = RELIABLE_MARKER;
    
    // Append payload
    const uint8_t* data = reinterpret_cast<const uint8_t*>(&pkt);
    buffer.insert(buffer.end(), data, data + sizeof(pkt));
    
    return buffer;
}

inline std::vector<uint8_t> packFiringCommand(bool fire) {
    std::vector<uint8_t> buffer;
    buffer.reserve(2 + sizeof(firingCommandPacket));
    
    buffer.push_back(USB_MAGIC_BYTE);
    buffer.push_back(ID_FIRING_COMMAND);
    
    firingCommandPacket pkt;
    pkt.fire_state = fire ? 1 : 0;
    pkt.__reliable = RELIABLE_MARKER;
    
    const uint8_t* data = reinterpret_cast<const uint8_t*>(&pkt);
    buffer.insert(buffer.end(), data, data + sizeof(pkt));
    
    return buffer;
}

// Optional: If you need to send chassis commands
inline std::vector<uint8_t> packChassisSpeedCommand(float vx, float vy, float vyaw) {
    std::vector<uint8_t> buffer;
    buffer.reserve(2 + sizeof(chassisSpeedCommandPacket));
    
    buffer.push_back(USB_MAGIC_BYTE);
    buffer.push_back(ID_CHASSIS_SPEED);
    
    chassisSpeedCommandPacket pkt;
    pkt.V_horz = vx;
    pkt.V_lat = vy;
    pkt.V_yaw = vyaw;
    pkt.__reliable = RELIABLE_MARKER;
    
    const uint8_t* data = reinterpret_cast<const uint8_t*>(&pkt);
    buffer.insert(buffer.end(), data, data + sizeof(pkt));
    
    return buffer;
}

// Helper to parse incoming packets from MCU
struct IncomingPacket {
    uint8_t id;
    std::vector<uint8_t> payload;
    
    bool isValid() const { return id != 0; }
};

inline IncomingPacket parseIncomingData(const uint8_t* data, size_t len) {
    IncomingPacket result{0, {}};
    
    if (len < 2) return result;
    if (data[0] != USB_MAGIC_BYTE) return result;
    
    result.id = data[1];
    
    // Payload starts at index 2
    size_t payload_len = len - 2;
    result.payload.assign(data + 2, data + len);
    
    return result;
}

// If you need to parse competition status from MCU
struct __attribute__((packed)) competitionStatusPacket {
    uint8_t game_progress;
    uint16_t time_left;
    uint8_t robot_id;
    uint16_t current_hp;
    
    // Team HPs
    uint16_t red_hero_hp;
    uint16_t red_standard_hp;
    uint16_t red_sentry_hp;
    
    uint16_t blue_hero_hp;
    uint16_t blue_standard_hp;
    uint16_t blue_sentry_hp;
    
    uint32_t __reliable;
};

} // namespace protocol