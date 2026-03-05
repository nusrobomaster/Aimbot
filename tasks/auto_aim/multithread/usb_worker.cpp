// tasks/auto_aim/multithread/usb_worker.cpp
#include "workers.hpp"
#include "usb_communication.h"
#include <chrono>
#include <thread>
#include <iostream>

USBWorker::USBWorker(SharedLatest &shared,
            SharedScalars &scalars,
            std::atomic<bool> &stop_flag,
            std::shared_ptr<calibur::USBCommunication> usb_comm)
    : shared_(shared), scalars_(scalars), stop_(stop_flag), usb_comm_(std::move(usb_comm)) {}

void USBWorker::operator()() {
    // Try to open USB device on ttyACM1
    if (!usb_comm_->isOpen()) {
        std::string port = "/dev/ttyACM1";
        int baudrate = 115200;  // Match your MCU baud rate
        
        std::cout << "[USBWorker] Opening " << port << " at " << baudrate << " baud" << std::endl;
        
        if (!usb_comm_->open(port, baudrate)) {
            std::cerr << "[USBWorker] Failed to open " << port << "!" << std::endl;
            std::cerr << "[USBWorker] Check: ls -la /dev/ttyACM*" << std::endl;
            std::cerr << "[USBWorker] Fix: sudo chmod 666 " << port << std::endl;
            return;
        }
    }
    
    // Flush any stale data
    usb_comm_->flush();
    
    const auto loop_interval = std::chrono::milliseconds(10); // 100Hz send rate
    
    // For debugging
    int send_count = 0;
    int no_target_count = 0;
    
    while (!stop_.load(std::memory_order_relaxed)) {
        auto start_time = std::chrono::steady_clock::now();
        
        // Get latest prediction data
        auto pred_ptr = std::atomic_load(&shared_.prediction_out);
        
        bool sent = false;
        
        if (pred_ptr && pred_ptr->valid) {
            // Valid target - send aimbot data
            // Your MCU expects: yaw, pitch, fire (0 or 1)
            sent = usb_comm_->sendAimbotData(
                pred_ptr->yaw,
                pred_ptr->pitch,
                pred_ptr->fire
            );
            
            no_target_count = 0;
            send_count++;
            
            // Print every 100th successful send
            if (send_count % 100 == 0) {
                std::cout << "[USBWorker] Sent aim: yaw=" << pred_ptr->yaw 
                         << " pitch=" << pred_ptr->pitch 
                         << " fire=" << pred_ptr->fire
                         << " dist=" << pred_ptr->distance << "m" << std::endl;
            }
        } else {
            // No target - send zeroed data (yaw=0, pitch=0, fire=0)
            // This keeps the gimbal still or in manual mode
            sent = usb_comm_->sendAimbotData(0.0f, 0.0f, false);
            
            no_target_count++;
            
            // Print periodically when no target
            if (no_target_count % 100 == 0) {
                std::cout << "[USBWorker] No target - sending zero command" << std::endl;
            }
        }
        
        if (!sent) {
            static int error_count = 0;
            error_count++;
            if (error_count % 100 == 0) {
                std::cerr << "[USBWorker] Failed to send data (" << error_count << " errors)" << std::endl;
            }
        }
        
        // Maintain constant frequency
        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = end_time - start_time;
        if (elapsed < loop_interval) {
            std::this_thread::sleep_for(loop_interval - elapsed);
        }
    }
    
    // Send zero command before exiting
    usb_comm_->sendAimbotData(0.0f, 0.0f, false);
    std::cout << "[USBWorker] Shutting down, sent zero command" << std::endl;
}

void USBWorker::process_usb_rx() {
    // Optional: Read any data from MCU
    uint8_t buffer[64];
    int bytes_read = usb_comm_->receiveData(buffer, sizeof(buffer), 0);
    
    if (bytes_read > 0) {
        // Process incoming data if your MCU sends anything back
        // For now, just print debug info occasionally
        static int rx_count = 0;
        if (++rx_count % 100 == 0) {
            std::cout << "[USBWorker] Received " << bytes_read << " bytes from MCU" << std::endl;
        }
    }
}