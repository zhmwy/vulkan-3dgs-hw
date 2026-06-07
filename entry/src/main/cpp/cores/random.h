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
    std::vector<std::vector<int>> trainOrders;
    int _range = -1;
    int _current_idx = 0;
    
    void initQueue() {
        std::shuffle(arr.begin(), arr.end(), gen);
        _current_idx = 0;
    }
    
public:
    void initialize(int range) {
        _range = range;
        constexpr uint32_t fixedSeed = 0;
        gen.seed(fixedSeed);             // Seed the generator
        dist = std::uniform_int_distribution<>(0, range - 1); // Set the distribution range
        arr.resize(range, 0);
        for(int i = 0; i < range; i++)
            arr[i] = i;
        std::shuffle(arr.begin(), arr.end(), gen);
    }
    
//     void initialize(std::vector<int> trainOrder) {
//         arr = trainOrder;
//         _range = -1;
//     }
//     void initialize(std::vector<std::vector<int>> _trainOrders) {
//         this->trainOrders = _trainOrders;
//         _range = trainOrders.size();
//         gen.seed(rd()); 
//         dist = std::uniform_int_distribution<>(0, _range - 1);
//     }
    
    int generate() {
        int k = arr[_current_idx % arr.size()];
        _current_idx++;
        if (_current_idx % arr.size() == 0) {
//             gen.seed(rd()); 
            std::shuffle(arr.begin(), arr.end(), gen);
        }
        return k;
        
//         if(_current_idx % _range == 0){
//             arr = trainOrders[dist(gen)];
//         }
//         int k = arr[_current_idx % _range];
//         _current_idx++;
//         return k;
    }

    
};
#endif //XENGINE_VULKAN_COMPUTE_CPP_RANDOM_H