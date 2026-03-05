// tasks/auto_aim/multithread/usb_communication.h
#pragma once

#include <string>
#include <atomic>
#include <cstdint>
#include <cstring>

namespace calibur {

constexpr uint8_t USB_MAGIC_BYTE = 0x7F;
constexpr uint8_t ID_GIMBAL_COMMAND = 12;
constexpr uint8_t ID_FIRING_COMMAND = 13;

// EXACTLY matching firmware structures
struct __attribute__((packed)) cvGimbalCommandPacket {
    float yaw;
    float pitch;
    uint16_t crc;  // 2-byte CRC at the end
};

struct __attribute__((packed)) firingCommandPacket {
    uint8_t fire_state;  // 0 or 1
    uint16_t crc;        // 2-byte CRC at the end
};

class USBCommunication {
public:
    USBCommunication();
    ~USBCommunication();

    bool open(const std::string& port = "/dev/ttyACM1", int baudrate = 115200);
    void close();
    bool isOpen() const;
    bool configure(int baudrate);
    bool writeData(const uint8_t* data, size_t length);
    int receiveData(uint8_t* buffer, size_t max_length, int timeout_ms = 0);
    void flush();
    
    bool sendGimbalCommand(float yaw, float pitch);
    bool sendFiringCommand(uint8_t fire);  // 0 or 1 only
    
private:
    // CRC16 implementation EXACTLY matching firmware
    static uint16_t crc16(const uint8_t* data, uint16_t size) {
        uint8_t x;
        uint16_t crc = 0xFFFF;

        for (uint16_t i = 0; i < size; i++) {
            x = (crc >> 8) ^ data[i];
            x ^= (x >> 4);
            crc = (crc << 8) ^ ((uint16_t)(x << 12)) ^ ((uint16_t)(x << 5)) ^ ((uint16_t)x);
        }
        return crc;
    }
    
    int fd_;
    std::atomic<bool> open_;
    int baudrate_;
};

} // namespace calibur