#include "utils.slang"

[[vk::binding(0)]] StructuredBuffer<uint32_t> mask;
[[vk::binding(1)]] StructuredBuffer<uint32_t> curPointsNum;

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

}
