#include "utils.slang"

[[vk::binding(0)]] StructuredBuffer<float> cov2d_array;
[[vk::binding(1)]] StructuredBuffer<uint32_t> visible_array;
[[vk::binding(2)]] RWStructuredBuffer<float> conic_array;

[[vk::binding(5)]] StructuredBuffer<int32_t> curPointsNum;

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
    if (idx >= curPointsNum[0] || visible_array[idx] == 0)
        return;

    float3 cov = {
        cov2d_array[3 * idx],
        cov2d_array[3 * idx + 1],
        cov2d_array[3 * idx + 2]
    };
    // float covx = cov2d_array[3 * idx];
    // float covy = cov2d_array[3 * idx + 1];
    // float covz = cov2d_array[3 * idx + 2];

    float det = (cov.x * cov.z - cov.y * cov.y);
    if (fabs(det) < EPS)
        return;

    float det_inv = 1.f / det;
    conic_array[3 * idx] = cov.z * det_inv;
    conic_array[3 * idx + 1] = -cov.y * det_inv;
    conic_array[3 * idx + 2] = cov.x * det_inv;
}