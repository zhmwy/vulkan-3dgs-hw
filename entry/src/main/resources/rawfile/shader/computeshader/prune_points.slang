#include "utils.slang"

[[vk::binding(0)]] StructuredBuffer<int32_t> max_radii2D_array;
[[vk::binding(1)]] RWStructuredBuffer<float> xyz_array;
[[vk::binding(2)]] RWStructuredBuffer<float> shs_array;
[[vk::binding(3)]] RWStructuredBuffer<float> opacity_array;
[[vk::binding(4)]] RWStructuredBuffer<float> scale_array;
[[vk::binding(5)]] RWStructuredBuffer<float> rotate_array;

[[vk::binding(6)]] RWStructuredBuffer<float> xyz_array_exp_avg;
[[vk::binding(7)]] RWStructuredBuffer<float> scale_array_exp_avg;
[[vk::binding(8)]] RWStructuredBuffer<float> rotate_array_exp_avg;
[[vk::binding(9)]] RWStructuredBuffer<float> feature_array_exp_avg;
[[vk::binding(10)]] RWStructuredBuffer<float> opacity_array_exp_avg;

[[vk::binding(11)]] RWStructuredBuffer<float> xyz_array_avg;
[[vk::binding(12)]] RWStructuredBuffer<float> scale_array_avg;
[[vk::binding(13)]] RWStructuredBuffer<float> rotate_array_avg;
[[vk::binding(14)]] RWStructuredBuffer<float> feature_array_avg;
[[vk::binding(15)]] RWStructuredBuffer<float> opacity_array_avg;
[[vk::binding(16)]] RWStructuredBuffer<int32_t> curPointsNum;

[[vk::push_constant]]
cbuffer PushConstants
{
    float xyz_size;
    float width;
    float height;
    float position_lr;
    float D;
    float extent;
    float _training_step;
};

bool isValid(int idx) {
    float opacity_act = 1 / (1 + exp(-opacity_array[idx]));
    if (opacity_act < min_opacity)
        return false;

    if (_training_step > 3000 && max_screen_size > 0) {
        if (max_radii2D_array[idx] > max_screen_size)
            return false;

        float3 scale_act = {
            exp(scale_array[t2i(idx, 0, 3)]),
            exp(scale_array[t2i(idx, 1, 3)]),
            exp(scale_array[t2i(idx, 2, 3)])
        };
        float scale_act_max = max(max(scale_act.x, scale_act.y), scale_act.z);

        if (scale_act_max > 0.1 * extent) {
            return false;
        }
    }
    return true;
}


[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint32_t idx = dispatchThreadID.x;
    if (idx > 0)
        return;

    int left = skybox_num;           // start from the original points
    int right = curPointsNum[0] - 1;
    while (left < right) {

        // 1. find the point to be pruned in the left;
        while (isValid(left)) {
            left++;
        }

        // 2. find the point to be swapped in the right;
        while (!isValid(right)) {
            resetGS(xyz_array, shs_array, int(D), opacity_array, scale_array, rotate_array, right);
            right--;
            curPointsNum[0]--;  // notice that, if a point in the rigth is invalid, the curPointsNum should decrease 
        }

        // 3. swap the left and right points
        if (left < right) {

            replaceGS(xyz_array, shs_array, DD, opacity_array, scale_array, rotate_array, right, left);
            replaceGS(xyz_array_avg, feature_array_avg, DD, opacity_array_avg, scale_array_avg, rotate_array_avg, right, left);
            replaceGS(xyz_array_exp_avg, feature_array_exp_avg, DD, opacity_array_exp_avg, scale_array_exp_avg, rotate_array_exp_avg, right, left);

            left++;
            right--;
            curPointsNum[0]--;
        }

    }
}

