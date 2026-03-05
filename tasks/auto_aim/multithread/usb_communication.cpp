#include "usb_communication.h"
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <linux/serial.h>
#include <cstring>
#include <iostream>
#include <iomanip>

namespace calibur {

USBCommunication::USBCommunication() 
    : fd_(-1), open_(false), baudrate_(115200) {
    std::cout << "[USB] Gimbal packet size: " << sizeof(cvGimbalCommandPacket) << " bytes" << std::endl;
    std::cout << "[USB] Firing packet size: " << sizeof(firingCommandPacket) << " bytes" << std::endl;
    std::cout << "[USB] Total with header - Gimbal: " << (2 + sizeof(cvGimbalCommandPacket)) << " bytes" << std::endl;
    std::cout << "[USB] Total with header - Firing: " << (2 + sizeof(firingCommandPacket)) << " bytes" << std::endl;
}

USBCommunication::~USBCommunication() { close(); }

bool USBCommunication::open(const std::string& port, int baudrate) {
    if (isOpen()) close();
    
    fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    if (fd_ < 0) return false;
    
    if (!configure(baudrate)) {
        ::close(fd_);
        fd_ = -1;
        return false;
    }
    
    open_ = true;
    std::cout << "[USB] Opened " << port << std::endl;
    return true;
}

void USBCommunication::close() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
    open_ = false;
}

bool USBCommunication::isOpen() const { return open_.load(); }

bool USBCommunication::configure(int baudrate) {
    if (fd_ < 0) return false;
    
    struct termios tty;
    memset(&tty, 0, sizeof(tty));
    if (tcgetattr(fd_, &tty) != 0) return false;
    
    speed_t speed = B115200;
    cfsetospeed(&tty, speed);
    cfsetispeed(&tty, speed);
    
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cflag |= CREAD | CLOCAL;
    
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY | ICRNL | INLCR);
    tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INPCK);
    tty.c_oflag &= ~OPOST;
    
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 1;
    
    return tcsetattr(fd_, TCSANOW, &tty) == 0;
}

bool USBCommunication::writeData(const uint8_t* data, size_t length) {
    if (!isOpen() || fd_ < 0) return false;
    
    // Debug: show exactly what's being sent
    std::cout << "[USB] TX (" << length << " bytes): ";
    for (size_t i = 0; i < length; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') 
                  << (int)data[i] << " ";
    }
    std::cout << std::dec << std::endl;
    
    ssize_t written = write(fd_, data, length);
    if (written == static_cast<ssize_t>(length)) {
        tcdrain(fd_);
        return true;
    }
    return false;
}

bool USBCommunication::sendGimbalCommand(float yaw, float pitch) {
    if (!isOpen()) return false;
    
    uint8_t buffer[2 + sizeof(cvGimbalCommandPacket)];
    
    buffer[0] = USB_MAGIC_BYTE;  // 0x7F
    buffer[1] = ID_GIMBAL_COMMAND;  // 12
    
    cvGimbalCommandPacket* pkt = reinterpret_cast<cvGimbalCommandPacket*>(buffer + 2);
    pkt->yaw = yaw;
    pkt->pitch = pitch - 0.04f;
    
    // Calculate CRC over yaw + pitch (8 bytes)
    uint16_t crc = crc16(reinterpret_cast<const uint8_t*>(pkt), sizeof(cvGimbalCommandPacket) - 2);
    pkt->crc = crc;
    
    std::cout << "[USB] Gimbal: yaw=" << yaw << ", pitch=" << pitch 
              << ", crc=0x" << std::hex << crc << std::dec << std::endl;
    
    return writeData(buffer, sizeof(buffer));
}

bool USBCommunication::sendFiringCommand(uint8_t fire) {
    if (!isOpen()) return false;
    
    uint8_t buffer[2 + sizeof(firingCommandPacket)];
    
    buffer[0] = USB_MAGIC_BYTE;  // 0x7F
    buffer[1] = ID_FIRING_COMMAND;  // 13
    
    firingCommandPacket* pkt = reinterpret_cast<firingCommandPacket*>(buffer + 2);
    pkt->fire_state = fire;
    
    // Calculate CRC over fire_state (1 byte)
    uint16_t crc = crc16(reinterpret_cast<const uint8_t*>(pkt), sizeof(firingCommandPacket) - 2);
    pkt->crc = crc;
    
    std::cout << "[USB] Fire: state=" << (int)fire 
              << ", crc=0x" << std::hex << crc << std::dec << std::endl;
    
    return writeData(buffer, sizeof(buffer));
}

int USBCommunication::receiveData(uint8_t* buffer, size_t max_length, int timeout_ms) {
    if (!isOpen() || fd_ < 0) return -1;
    return read(fd_, buffer, max_length);
}

void USBCommunication::flush() {
    if (fd_ >= 0) tcflush(fd_, TCIOFLUSH);
}

} // namespace calibur