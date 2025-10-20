#include "utils.slang"

[[vk::binding(0)]] StructuredBuffer<float> xyz_gradient_accum;
[[vk::binding(1)]] StructuredBuffer<int32_t> denom;
[[vk::binding(2)]] StructuredBuffer<int32_t> curPointsNum;
[[vk::binding(3)]] RWStructuredBuffer<float> grads;
[[vk::binding(4)]] RWStructuredBuffer<uint32_t> frozen;
[[vk::binding(5)]] RWStructuredBuffer<int> unsorted_idx_array;
[[vk::binding(6)]] RWStructuredBuffer<uint32_t> grads_int;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint32_t idx = dispatchThreadID.x;
    if (idx >= curPointsNum[0])
        return;

    if (denom[idx] == 0)
        grads[idx] = 0;
    else
        grads[idx] = xyz_gradient_accum[idx] / denom[idx];
    
    // if (grads[idx] < 0.000001)
    //     frozen[idx] = 1;

    // grads[idx] = 1 - grads[idx];
    unsorted_idx_array[idx] = idx;
    grads_int[idx] = reinterpret<uint32_t>((1 - grads[idx]) * 10000000);
}

