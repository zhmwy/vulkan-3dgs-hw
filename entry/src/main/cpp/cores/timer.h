//
// Created on 2024/12/5.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef XENGINE_VULKAN_COMPUTE_CPP_TIMER_H
#define XENGINE_VULKAN_COMPUTE_CPP_TIMER_H
#include <iostream>
#include <chrono>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start;

public:
    // Constructor to initialize 'start'
    Timer() : start(std::chrono::high_resolution_clock::now()), last_time(0) {}

    // Method to return elapsed time in milliseconds
    long long elapsed() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
    void upd_last_time(long long time) {
        last_time = time;
    }

    long long last_time;
};
#endif //XENGINE_VULKAN_COMPUTE_CPP_TIMER_H
