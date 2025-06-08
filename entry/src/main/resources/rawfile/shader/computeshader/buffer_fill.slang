#include "utils.slang"

[[vk::binding(0)]] RWStructuredBuffer<uint32_t> InputBuffer;
[[vk::binding(1)]] StructuredBuffer<int32_t> curPointsNum;
// [[vk::binding(2)]] StructuredBuffer<float> opacity_array;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint xyz_size;
    uint width;
    uint height;
    uint training_step;
};


[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint32_t idx = dispatchThreadID.x;
    if (idx.x >= curPointsNum[0])
        return;

    InputBuffer[idx] = 1;
    // if (training_step >= 501 && (training_step - 1) % 100 == 0) {
    //     float opacity_act = 1 / (1 + exp(-opacity_array[idx]));
    //     if (opacity_act < min_opacity) {
    //         InputBuffer[idx] = 0;
    //     }
    //     else {
    //         InputBuffer[idx] = 1;
    //     }
    // }
    // else {
    //     InputBuffer[idx] = 1;
    // }
}
