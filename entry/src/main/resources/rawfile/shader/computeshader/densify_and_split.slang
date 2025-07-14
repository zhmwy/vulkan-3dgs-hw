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

float rand(float2 seed) {
    return frac(sin(dot(seed, float2(12.9898, 78.233))) * 43758.5453);
}

// Box-Muller mean = 0, std = 1
float gaussianRandom(float2 seed) {
    float u1 = rand(seed);
    float u2 = rand(seed + float2(1.0, 0.0));
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265359 * u2);
}

float3 normalRandomXYZ(float3 xyz, float3 std) {
    float3 noise;
    noise.x = std.x * gaussianRandom(float2(xyz.y, xyz.z));
    noise.y = std.y * gaussianRandom(float2(xyz.x, xyz.z));
    noise.z = std.z * gaussianRandom(float2(xyz.x, xyz.y));
    return noise;
}

[[vk::binding(0)]] StructuredBuffer<int32_t> toDensifyIdx_array;
[[vk::binding(1)]] RWStructuredBuffer<float> xyz_array;
[[vk::binding(2)]] RWStructuredBuffer<float> shs_array;
[[vk::binding(3)]] RWStructuredBuffer<float> opacity_array;
[[vk::binding(4)]] RWStructuredBuffer<float> scale_array;
[[vk::binding(5)]] RWStructuredBuffer<float> rotate_array;
[[vk::binding(6)]] RWByteAddressBuffer curPointsNum;

[[vk::binding(7)]] RWStructuredBuffer<float> xyz_array_exp_avg;
[[vk::binding(8)]] RWStructuredBuffer<float> shs_array_exp_avg;
[[vk::binding(9)]] RWStructuredBuffer<float> opacity_array_exp_avg;
[[vk::binding(10)]] RWStructuredBuffer<float> scale_array_exp_avg;
[[vk::binding(11)]] RWStructuredBuffer<float> rotate_array_exp_avg;

[[vk::binding(12)]] RWStructuredBuffer<float> xyz_array_avg;
[[vk::binding(13)]] RWStructuredBuffer<float> shs_array_avg;
[[vk::binding(14)]] RWStructuredBuffer<float> opacity_array_avg;
[[vk::binding(15)]] RWStructuredBuffer<float> scale_array_avg;
[[vk::binding(16)]] RWStructuredBuffer<float> rotate_array_avg;

[[vk::binding(17)]] RWStructuredBuffer<float> grads;

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
    if (idx >= max(xyz_size * 0.05, 10000.0))
        return;

    uint32_t toDensifyIdx_reg = toDensifyIdx_array[idx];
    if (1 / (1 + exp(-opacity_array[toDensifyIdx_reg])) < min_opacity) {
        return;
    }

    if (grads[toDensifyIdx_reg] < densify_grad_threshold && training_step > 500)
        return;

    float3 scale_act = {
        exp(scale_array[t2i(toDensifyIdx_reg, 0, 3)]),
        exp(scale_array[t2i(toDensifyIdx_reg, 1, 3)]),
        exp(scale_array[t2i(toDensifyIdx_reg, 2, 3)])
    };
    float scale_act_max = max(max(scale_act.x, scale_act.y), scale_act.z);
    if (scale_act_max <= percent_dense * scene_extent)
        return;

    
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

    float3 old_xyz = {
        xyz_array[t2i(toDensifyIdx_reg, 0, 3)],
        xyz_array[t2i(toDensifyIdx_reg, 1, 3)],
        xyz_array[t2i(toDensifyIdx_reg, 2, 3)],
    };

    // float3 sample = normalRandomXYZ(old_xyz, scale_act);
    // float3 sample2 = normalRandomXYZ(old_xyz + float3(0.1, 0.2, 0.3), scale_act);

    float3 sample = { 0, 0, 0 };
    if (scale_act.x >= scale_act.y && scale_act.x >= scale_act.z)
        sample[0] = scale_act.x * 0.6636;
    else if (scale_act.y >= scale_act.x && scale_act.y >= scale_act.z)
        sample[1] = scale_act.y * 0.6636;
    else
        sample[2] = scale_act.z * 0.6636;

    float3 t1 = {
        R[0][0] * sample[0] + R[0][1] * sample[1] + R[0][2] * sample[2],
        R[1][0] * sample[0] + R[1][1] * sample[1] + R[1][2] * sample[2],
        R[2][0] * sample[0] + R[2][1] * sample[1] + R[2][2] * sample[2]
    };
    float3 t2 = -t1;

    // float3 t2 = {
    //     R[0][0] * sample2[0] + R[0][1] * sample2[1] + R[0][2] * sample2[2],
    //     R[1][0] * sample2[0] + R[1][1] * sample2[1] + R[1][2] * sample2[2],
    //     R[2][0] * sample2[0] + R[2][1] * sample2[1] + R[2][2] * sample2[2]
    // };

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

    // rotate_array[4 * toInsertIdx + 0] = 200;
    // rotate_array[4 * toInsertIdx + 1] = idx * 1.0;
    // rotate_array[4 * toInsertIdx + 2] = toDensifyIdx_reg * 1.0;
    // rotate_array[4 * toInsertIdx + 3] = toInsertIdx * 1.0;

    // return;

    xyz_array[t2i(toInsertIdx, 0, 3)] = t2[0] + old_xyz.x;
    xyz_array[t2i(toInsertIdx, 1, 3)] = t2[1] + old_xyz.y;
    xyz_array[t2i(toInsertIdx, 2, 3)] = t2[2] + old_xyz.z;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < DD; j++) {
            shs_array[t2i(toInsertIdx, i, j, 3, DD)] = shs_array[t2i(toDensifyIdx_reg, i, j, 3, DD)];
        }
    }

    opacity_array[toInsertIdx] = opacity_array[toDensifyIdx_reg];

    scale_array[t2i(toInsertIdx, 0, 3)] = new_scale12[0];
    scale_array[t2i(toInsertIdx, 1, 3)] = new_scale12[1];
    scale_array[t2i(toInsertIdx, 2, 3)] = new_scale12[2];

    rotate_array[t2i(toInsertIdx, 0, 4)] = rotate_array[t2i(toDensifyIdx_reg, 0, 4)];
    rotate_array[t2i(toInsertIdx, 1, 4)] = rotate_array[t2i(toDensifyIdx_reg, 1, 4)];
    rotate_array[t2i(toInsertIdx, 2, 4)] = rotate_array[t2i(toDensifyIdx_reg, 2, 4)];
    rotate_array[t2i(toInsertIdx, 3, 4)] = rotate_array[t2i(toDensifyIdx_reg, 3, 4)];

    // reset exp_avg and exp_avg_sq
    resetGS(xyz_array_exp_avg, shs_array_exp_avg, DD, opacity_array_exp_avg, scale_array_exp_avg, rotate_array_exp_avg, toDensifyIdx_reg);
    resetGS(xyz_array_avg, shs_array_avg, DD, opacity_array_avg, scale_array_avg, rotate_array_avg, toDensifyIdx_reg);
}