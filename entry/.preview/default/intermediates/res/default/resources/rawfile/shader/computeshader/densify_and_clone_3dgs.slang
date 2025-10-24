#include "utils.slang"

[[vk::binding(0)]] StructuredBuffer<int32_t> toDensifyIdx_array;
[[vk::binding(1)]] RWStructuredBuffer<float> xyz_array;
[[vk::binding(2)]] RWStructuredBuffer<float> shs_array;
[[vk::binding(3)]] RWStructuredBuffer<float> opacity_array;
[[vk::binding(4)]] RWStructuredBuffer<float> scale_array;
[[vk::binding(5)]] RWStructuredBuffer<float> rotate_array;
[[vk::binding(6)]] RWByteAddressBuffer curPointsNum;
[[vk::binding(7)]] StructuredBuffer<float> grads;

[[vk::push_constant]]
cbuffer PushConstants
{
    float xyz_size;
    float width;
    float height;
    float position_lr;
    float D;
    float scene_extent;
    float training_step;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint32_t idx = dispatchThreadID.x;
    if (idx >= xyz_size * 0.05) 
        return;

    if (grads[idx] > 0.995)
        return;

    uint32_t toDensifyIdx_reg = toDensifyIdx_array[idx];

    float3 scale_act = {
        exp(scale_array[t2i(toDensifyIdx_reg, 0, 3)]),
        exp(scale_array[t2i(toDensifyIdx_reg, 1, 3)]),
        exp(scale_array[t2i(toDensifyIdx_reg, 2, 3)])
    };

    float scale_act_max = max(max(scale_act.x, scale_act.y), scale_act.z);

    if (scale_act_max > percent_dense * scene_extent)
        return;

    int32_t toInsertIdx = 0;
    curPointsNum.InterlockedAdd(0, 1, toInsertIdx);

    // xyz_array[toInsertIdx * 3 + 0] = 100.0;
    // xyz_array[toInsertIdx * 3 + 1] = toInsertIdx;
    // xyz_array[toInsertIdx * 3 + 2] = idx;
    // return;

    replaceGS(xyz_array, shs_array, DD, opacity_array, scale_array, rotate_array, toDensifyIdx_reg, toInsertIdx);
}