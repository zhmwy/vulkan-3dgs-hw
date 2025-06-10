#include "utils.slang"

[[vk::binding(0)]] StructuredBuffer<int32_t> max_radii2D_array;
[[vk::binding(1)]] RWStructuredBuffer<float> xyz_array;
[[vk::binding(2)]] RWStructuredBuffer<float> shs_array;
[[vk::binding(3)]] RWStructuredBuffer<float> opacity_array;
[[vk::binding(4)]] RWStructuredBuffer<float> scale_array;
[[vk::binding(5)]] RWStructuredBuffer<float> rotate_array;
[[vk::binding(6)]] RWStructuredBuffer<float> xyz_array_exp_avg;
[[vk::binding(7)]] RWStructuredBuffer<float> scale_array_exp_avg;
[[vk::binding(8)]] RWStructuredBuffer<float4> rotate_array_exp_avg;
[[vk::binding(9)]] RWStructuredBuffer<float> feature_array_exp_avg;
[[vk::binding(10)]] RWStructuredBuffer<float> opacity_array_exp_avg;

[[vk::binding(11)]] RWStructuredBuffer<float> xyz_array_avg;
[[vk::binding(12)]] RWStructuredBuffer<float> scale_array_avg;
[[vk::binding(13)]] RWStructuredBuffer<float4> rotate_array_avg;
[[vk::binding(14)]] RWStructuredBuffer<float> feature_array_avg;
[[vk::binding(15)]] RWStructuredBuffer<float> opacity_array_avg;
[[vk::binding(16)]] RWByteAddressBuffer curPointsNum;

// [[vk::push_constant]]
// cbuffer PushConstants
// {
//     int max_screen_size;        // 这里如果不要一定设置为-1, 不能是0
//     float min_opacity;
//     float scene_extent;
//     int oriPointsNum;           // 这个函数会用原子加改变curPointsNum, 记录没有改变时候的curPointsNum
//     int D;
// };

// [[vk::push_constant]]
// cbuffer PushConstants
// {
//     uint xyz_size;
//     uint width;
//     uint height;
//     uint temp;
// };

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
    if (idx >= xyz_size)
        return;

    float opacity_act = 1 / (1 + exp(-opacity_array[idx]));
    bool prune_mask = opacity_act < min_opacity;

    if (!prune_mask && max_screen_size >= 0) {

        prune_mask = max_radii2D_array[idx] > max_screen_size;

        if (!prune_mask) {
            float3 scale_act = {
                exp(scale_array[t2i(idx, 0, 3)]),
                exp(scale_array[t2i(idx, 1, 3)]),
                exp(scale_array[t2i(idx, 2, 3)])
            };
            float scale_act_max = max(max(scale_act.x, scale_act.y), scale_act.z);
            prune_mask = scale_act_max > 0.1 * scene_extent;
        }
    }

    if (!prune_mask)
        return;

    int32_t oldvalue = 0;
    curPointsNum.InterlockedAdd(0, -1, oldvalue);

    if (oldvalue - 1 < idx) {
        replaceGS(
            xyz_array,
            shs_array,
            D,
            opacity_array,
            scale_array,
            rotate_array,
            xyz_array_exp_avg,
            scale_array_exp_avg,
            feature_array_exp_avg, // This corresponds to shs_exp_avg
            rotate_array_exp_avg,
            opacity_array_exp_avg,
            xyz_array_avg,
            scale_array_avg,
            feature_array_avg, // This corresponds to shs_avg
            rotate_array_avg,
            opacity_array_avg,
            oldvalue - 1,
            idx
        );

        // TODO:
        // 是否需要把GS所有参数归0
        resetGS(xyz_array, shs_array, D, opacity_array, scale_array, rotate_array, oldvalue - 1);
    }

}

