#include "utils.slang"

[[vk::binding(0)]] StructuredBuffer<float> opacity_act_array;
[[vk::binding(1)]] StructuredBuffer<uint32_t> curPointsNum;
[[vk::binding(2)]] RWStructuredBuffer<uint32_t> mask;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint xyz_size;
    uint width;
    uint height;
    uint temp;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint32_t idx = dispatchThreadID.x;
    if (idx >= curPointsNum[0])
        return;

    if (opacity_act_array[idx] < min_opacity)
        mask[idx] = 0;
    else
        mask[idx] = 1;
}