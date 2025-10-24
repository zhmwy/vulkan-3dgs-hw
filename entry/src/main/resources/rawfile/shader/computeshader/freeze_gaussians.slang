#include "utils.slang"

[[vk::binding(0)]] RWStructuredBuffer<uint32_t> frozen;
[[vk::binding(1)]] StructuredBuffer<int32_t> curPointsNum;
[[vk::binding(2)]] StructuredBuffer<uint32_t> xyz_mean_grads;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint32_t idx = dispatchThreadID.x;
    if (idx >= curPointsNum[0])
        return;

    if (xyz_mean_grads[idx] < 0.000001 && idx>10000)
        frozen[idx] = 1;
}