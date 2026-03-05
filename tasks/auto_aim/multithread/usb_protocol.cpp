// tasks/auto_aim/multithread/usb_protocol.hpp
#pragma once

#include <cstdint>
#include <array>
#include <cstring>

namespace protocol {

// Magic bytes for packet validation
constexpr uint8_t PACKET_START_BYTE = 0xAA;
constexpr uint8_t PACKET_END_BYTE = 0x55;

// Packet types
enum class PacketType : uint8_t {
    HEARTBEAT = 0x01,
    AIMBOT_DATA = 0x02,
    GIMBAL_STATUS = 0x03,
    SHOOTER_STATUS = 0x04,
    TARGET_INFO = 0x05,
    DEBUG_DATA = 0x06,
    CONFIG_REQUEST = 0x07,
    CONFIG_RESPONSE = 0x08,
    CALIBRATION = 0x09,
    SYSTEM_STATUS = 0x0A
};

// Gimbal modes
enum class GimbalMode : uint8_t {
    MANUAL = 0x00,
    AUTO_AIM = 0x01,
    CALIBRATION = 0x02,
    FAULT = 0x03
};

// Shooter modes
enum class ShooterMode : uint8_t {
    IDLE = 0x00,
    READY = 0x01,
    FIRING = 0x02,
    BLOCKED = 0x03,
    LOW_SPEED = 0x04
};

// Packet header (8 bytes)
struct PacketHeader {
    uint8_t start_byte = PACKET_START_BYTE;
    uint8_t version = 0x01;
    PacketType type;
    uint8_t sequence_num;
    uint16_t payload_length;
    uint8_t checksum;
    uint8_t reserved = 0x00;
} __attribute__((packed));

// Heartbeat packet (simple)
struct HeartbeatPacket {
    uint32_t timestamp_ms;
    GimbalMode mode;
    ShooterMode shooter_mode;
    uint8_t system_health; // 0-100%
} __attribute__((packed));

// AimBot data packet (Jetson -> MCU)
struct AimbotDataPacket {
    float yaw_angle;        // radians, -π to π
    float pitch_angle;      // radians, -π/2 to π/2
    uint8_t fire_command;   // 0 = no fire, 1 = fire
    uint8_t target_locked;  // 0 = no target, 1 = locked
    uint8_t target_id;      // 1-6 for robots, 7 for base, 8 for outpost
    uint8_t target_type;    // 0 = unknown, 1 = hero, 2 = engineer, etc.
    float target_distance;  // meters
    float bullet_speed;     // m/s (requested)
    uint32_t timestamp_us;  // microseconds
} __attribute__((packed));

// Gimbal status packet (MCU -> Jetson)
struct GimbalStatusPacket {
    float current_yaw;      // radians
    float current_pitch;    // radians
    float target_yaw;       // radians
    float target_pitch;     // radians
    float yaw_speed;        // rad/s
    float pitch_speed;      // rad/s
    GimbalMode mode;
    uint8_t yaw_limit;      // 0 = not limited, 1 = at limit
    uint8_t pitch_limit;    // 0 = not limited, 1 = at limit
    uint8_t calibrated;     // 0 = not calibrated, 1 = calibrated
} __attribute__((packed));

// Shooter status packet (MCU -> Jetson)
struct ShooterStatusPacket {
    float left_wheel_speed;    // rpm
    float right_wheel_speed;   // rpm
    float target_speed;        // rpm
    ShooterMode mode;
    uint8_t block_detected;    // 0 = no block, 1 = blocked
    uint8_t remaining_projectiles; // 0-50
    uint8_t temperature;       // Celsius
} __attribute__((packed));

// Target info packet (Jetson -> MCU) - for debugging/display
struct TargetInfoPacket {
    float position_x;      // meters in world frame
    float position_y;
    float position_z;
    float velocity_x;      // m/s
    float velocity_y;
    float velocity_z;
    float confidence;      // 0.0 - 1.0
    uint8_t target_id;
    uint8_t armor_count;   // 2-4
} __attribute__((packed));

// Debug data packet (bidirectional)
struct DebugDataPacket {
    float p_gain;
    float i_gain;
    float d_gain;
    float feedforward;
    uint8_t debug_mode;
    float debug_value_1;
    float debug_value_2;
} __attribute__((packed));

// System status packet (MCU -> Jetson)
struct SystemStatusPacket {
    float voltage;         // V
    float current;         // A
    float temperature;     // C
    uint8_t error_code;
    uint8_t warning_code;
    uint32_t uptime_ms;
} __attribute__((packed));

// Main packet structure for sending/receiving
struct Packet {
    PacketHeader header;
    union {
        HeartbeatPacket heartbeat;
        AimbotDataPacket aimbot;
        GimbalStatusPacket gimbal;
        ShooterStatusPacket shooter;
        TargetInfoPacket target;
        DebugDataPacket debug;
        SystemStatusPacket system;
        uint8_t raw_data[256];
    } payload;
    
    // Calculate total packet size
    size_t totalSize() const {
        return sizeof(PacketHeader) + header.payload_length;
    }
    
    // Verify packet integrity
    bool verify() const {
        if (header.start_byte != PACKET_START_BYTE)
            return false;
        
        uint8_t calculated_checksum = calculateChecksum();
        return calculated_checksum == header.checksum;
    }
    
    // Calculate checksum (XOR of all bytes except checksum field)
    uint8_t calculateChecksum() const {
        const uint8_t* data = reinterpret_cast<const uint8_t*>(this);
        uint8_t checksum = 0;
        
        // Skip start_byte and checksum field
        for (size_t i = 1; i < sizeof(PacketHeader); i++) {
            if (i != offsetof(PacketHeader, checksum)) {
                checksum ^= data[i];
            }
        }
        
        // XOR payload
        for (size_t i = 0; i < header.payload_length; i++) {
            checksum ^= reinterpret_cast<const uint8_t*>(&payload)[i];
        }
        
        return checksum;
    }
    
    // Prepare packet for sending
    void finalize() {
        header.checksum = calculateChecksum();
    }
} __attribute__((packed));

// Helper functions
inline Packet createAimbotPacket(float yaw, float pitch, bool fire, 
                                bool locked, int target_id, int target_type,
                                float distance, float bullet_speed) {
    Packet pkt;
    pkt.header = {
        .start_byte = PACKET_START_BYTE,
        .version = 0x01,
        .type = PacketType::AIMBOT_DATA,
        .sequence_num = 0,
        .payload_length = sizeof(AimbotDataPacket),
        .checksum = 0,
        .reserved = 0
    };
    
    pkt.payload.aimbot = {
        .yaw_angle = yaw,
        .pitch_angle = pitch,
        .fire_command = fire ? 1 : 0,
        .target_locked = locked ? 1 : 0,
        .target_id = static_cast<uint8_t>(target_id),
        .target_type = static_cast<uint8_t>(target_type),
        .target_distance = distance,
        .bullet_speed = bullet_speed,
        .timestamp_us = 0  // Will be filled before sending
    };
    
    pkt.finalize();
    return pkt;
}

} // namespace protocol