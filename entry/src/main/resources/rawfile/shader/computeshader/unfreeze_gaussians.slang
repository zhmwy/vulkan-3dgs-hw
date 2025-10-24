#include "utils.slang"

[[vk::binding(0)]] RWStructuredBuffer<uint32_t> frozen;
[[vk::binding(1)]] StructuredBuffer<int32_t> curPointsNum;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint32_t idx = dispatchThreadID.x;
    if (idx >= curPointsNum[0])
        return;

    // Unfreeze all gaussians to allow fine-tuning in final phase
    frozen[idx] = 0;
}