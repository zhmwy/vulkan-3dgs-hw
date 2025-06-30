#include "utils.slang"

[[vk::binding(0)]] StructuredBuffer<float> cov2d_array;
[[vk::binding(1)]] StructuredBuffer<uint32_t> visible_array;
[[vk::binding(3)]] StructuredBuffer<float> dL_dconic_array;
[[vk::binding(4)]] RWStructuredBuffer<float> dL_dcov2d_array;
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

    float3 cov = float3(
        cov2d_array[idx * 3 + 0],
        cov2d_array[idx * 3 + 1],
        cov2d_array[idx * 3 + 2]
    );
    float det = (cov.x * cov.z - cov.y * cov.y);
    if (fabs(det) < EPS)
        return;

    float nom = 1.0 / det / det;
    float3 dL_dconic_reg = float3(
        dL_dconic_array[idx * 3 + 0],
        dL_dconic_array[idx * 3 + 1],
        dL_dconic_array[idx * 3 + 2]
    );
    dL_dcov2d_array[idx * 3 + 0] = nom * (dL_dconic_reg.x * (-cov.z) * cov.z +
                                          dL_dconic_reg.y * cov.y * cov.z +
                                          dL_dconic_reg.z * (det - cov.x * cov.z));
    dL_dcov2d_array[idx * 3 + 1] = nom * (dL_dconic_reg.x * 2 * cov.z * cov.y +
                                          dL_dconic_reg.y * (-2 * cov.y * cov.y - det) +
                                          dL_dconic_reg.z * 2 * cov.x * cov.y);
    dL_dcov2d_array[idx * 3 + 2] = nom * (dL_dconic_reg.x * (det - cov.x * cov.z) +
                                          dL_dconic_reg.y * cov.x * cov.y +
                                          dL_dconic_reg.z * (-cov.x * cov.x));
}
