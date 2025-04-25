#include "utils.slang"

float3x3 unit_quaternion_to_rotmatrix(float4 rotate) {
    float w = rotate[0];
    float x = rotate[1];
    float y = rotate[2];
    float z = rotate[3];

    return float3x3(
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)
    );
}

[[vk::binding(0)]] StructuredBuffer<int32_t> toDensifyIdx_array;
[[vk::binding(1)]] RWStructuredBuffer<float> xyz_array;
[[vk::binding(2)]] RWStructuredBuffer<float4> shs_array;
[[vk::binding(3)]] RWStructuredBuffer<float> opacity_array;
[[vk::binding(4)]] RWStructuredBuffer<float> scale_array;
[[vk::binding(5)]] RWStructuredBuffer<float> rotate_array;
[[vk::binding(6)]] RWByteAddressBuffer curPointsNum;

[[vk::push_constant]]
cbuffer PushConstants
{
    float xyz_size;
    float width;
    float height;
    float position_lr;
    float D;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint32_t idx = dispatchThreadID.x;
    if (idx >= xyz_size * 0.05)
        return;

    uint32_t toDensifyIdx_reg = toDensifyIdx_array[idx];

    float3 scale_act = {
        exp(scale_array[t2i(toDensifyIdx_reg, 0, 3)]),
        exp(scale_array[t2i(toDensifyIdx_reg, 1, 3)]),
        exp(scale_array[t2i(toDensifyIdx_reg, 2, 3)])
    };
    float scale_act_max = max(max(scale_act.x, scale_act.y), scale_act.z);
    if (scale_act_max <= percent_dense * scene_extent)
        return;

    float3 sample = { 0, 0, 0 };
    if (scale_act.x >= scale_act.y && scale_act.x >= scale_act.z)
        sample[0] = scale_act.x * 0.6636;
    else if (scale_act.y >= scale_act.x && scale_act.y >= scale_act.z)
        sample[1] = scale_act.y * 0.6636;
    else
        sample[2] = scale_act.z * 0.6636;

    float4 rotate_reg = {
        rotate_array[t2i(toDensifyIdx_reg, 0, 4)],
        rotate_array[t2i(toDensifyIdx_reg, 1, 4)],
        rotate_array[t2i(toDensifyIdx_reg, 2, 4)],
        rotate_array[t2i(toDensifyIdx_reg, 3, 4)]
    };
    float norm_rotate = sqrt(rotate_reg[0] * rotate_reg[0] +
                             rotate_reg[1] * rotate_reg[1] +
                             rotate_reg[2] * rotate_reg[2] +
                             rotate_reg[3] * rotate_reg[3]);
    float4 rotate_act = rotate_reg / norm_rotate;
    float3x3 R = unit_quaternion_to_rotmatrix(rotate_act);

    float3 t1 = {
        R[0][0] * sample[0] + R[0][1] * sample[1] + R[0][2] * sample[2],
        R[1][0] * sample[0] + R[1][1] * sample[1] + R[1][2] * sample[2],
        R[2][0] * sample[0] + R[2][1] * sample[1] + R[2][2] * sample[2]
    };

    // float3 t2 = -t1;

    float3 old_xyz = {
        xyz_array[t2i(toDensifyIdx_reg, 0, 3)],
        xyz_array[t2i(toDensifyIdx_reg, 1, 3)],
        xyz_array[t2i(toDensifyIdx_reg, 2, 3)],
    };

    float3 new_scale12 = log(scale_act / 1.6);

    // rotatation unchanged
    // SH unchanged
    // opacity unchanged

    // 1. 替代原来位置
    xyz_array[t2i(toDensifyIdx_reg, 0, 3)] = t1[0] + old_xyz.x;
    xyz_array[t2i(toDensifyIdx_reg, 1, 3)] = t1[1] + old_xyz.y;
    xyz_array[t2i(toDensifyIdx_reg, 2, 3)] = t1[2] + old_xyz.z;

    scale_array[t2i(toDensifyIdx_reg, 0, 3)] = new_scale12[0];
    scale_array[t2i(toDensifyIdx_reg, 1, 3)] = new_scale12[1];
    scale_array[t2i(toDensifyIdx_reg, 2, 3)] = new_scale12[2];

    // 2. 添加到后面
    int32_t toInsertIdx = 0;
    curPointsNum.InterlockedAdd(0, 1, toInsertIdx);

    xyz_array[t2i(toInsertIdx, 0, 3)] = -t1[0] + old_xyz.x;
    xyz_array[t2i(toInsertIdx, 1, 3)] = -t1[1] + old_xyz.y;
    xyz_array[t2i(toInsertIdx, 2, 3)] = -t1[2] + old_xyz.z;

    for (int i = 0; i < 3; i++) {
        shs_array[toInsertIdx*3 + i] = shs_array[toDensifyIdx_reg*3 + i];
    }

    opacity_array[toInsertIdx] = opacity_array[toDensifyIdx_reg];

    scale_array[t2i(toInsertIdx, 0, 3)] = new_scale12[0];
    scale_array[t2i(toInsertIdx, 1, 3)] = new_scale12[1];
    scale_array[t2i(toInsertIdx, 2, 3)] = new_scale12[2];

    rotate_array[t2i(toInsertIdx, 0, 4)] = rotate_array[t2i(toDensifyIdx_reg, 0, 4)];
    rotate_array[t2i(toInsertIdx, 1, 4)] = rotate_array[t2i(toDensifyIdx_reg, 1, 4)];
    rotate_array[t2i(toInsertIdx, 2, 4)] = rotate_array[t2i(toDensifyIdx_reg, 2, 4)];
    rotate_array[t2i(toInsertIdx, 3, 4)] = rotate_array[t2i(toDensifyIdx_reg, 3, 4)];

}