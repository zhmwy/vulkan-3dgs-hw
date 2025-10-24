#include "utils.slang"

[[vk::binding(0)]] StructuredBuffer<int32_t> radius;
[[vk::binding(1)]] StructuredBuffer<float> dL_duvd_array;
[[vk::binding(2)]] StructuredBuffer<int32_t> curPointsNum;
[[vk::binding(3)]] RWStructuredBuffer<int32_t> max_radii2D;
[[vk::binding(4)]] RWStructuredBuffer<float> xyz_gradient_accum;
[[vk::binding(5)]] RWStructuredBuffer<int32_t> denom;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint xyz_size;
    uint W;
    uint H;
    uint temp;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint32_t idx = dispatchThreadID.x;
    if (idx.x >= xyz_size || radius[idx] <= 0)
        return;

    max_radii2D[idx] = max(max_radii2D[idx], radius[idx]);

    // float2 dL_dndc_reg = dL_dndc_array[idx];
    float2 dL_dndc_reg = float2(
        dL_duvd_array[t2i(idx, 0, 3)] * 0.5 * W,
        dL_duvd_array[t2i(idx, 1, 3)] * 0.5 * H
    );

    xyz_gradient_accum[idx] = xyz_gradient_accum[idx] + sqrt(dL_dndc_reg[0] * dL_dndc_reg[0] + dL_dndc_reg[1] * dL_dndc_reg[1]);
    // xyz_gradient_accum[idx] = 1.0;

    denom[idx] = denom[idx] + 1;

}
