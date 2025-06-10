//
// Created on 2024/12/5.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef XENGINE_VULKAN_COMPUTE_CPP_RANDOM_H
#define XENGINE_VULKAN_COMPUTE_CPP_RANDOM_H
#include <random>
#include <vector>
#include <queue>

class RandomGenerator {
private:
    std::random_device rd;          // Seed generator
    std::mt19937 gen;               // Mersenne Twister RNG
    std::uniform_int_distribution<> dist;
    std::vector<int> arr;
    int _range = -1;
    int _current_idx = 0;
    
    void initQueue() {
        std::shuffle(arr.begin(), arr.end(), gen);
        _current_idx = 0;
    }
    
public:
    void initialize(int range) {
        _range = range;
        gen.seed(rd());             // Seed the generator
        dist = std::uniform_int_distribution<>(0, range - 1); // Set the distribution range
        arr.resize(range, 0);
        for(int i = 0; i < range; i++)
            arr[i] = i;
        std::shuffle(arr.begin(), arr.end(), gen);
    }
    
    int generate() {
        if(_current_idx == _range){
            this->initQueue();
        }
        int k = arr[_current_idx];
        _current_idx++;
        return k;
//         return dist(gen);           // Generate a random number
    }

    
};
#endif //XENGINE_VULKAN_COMPUTE_CPP_RANDOM_H