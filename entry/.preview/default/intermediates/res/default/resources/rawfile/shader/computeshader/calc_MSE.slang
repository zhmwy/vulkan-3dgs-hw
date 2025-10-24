# include "utils.slang"
[[vk::binding(0)]] StructuredBuffer<float> input_image_array;
[[vk::binding(1)]] StructuredBuffer<float> rfeat_array;
[[vk::binding(2)]] RWByteAddressBuffer psnr_sum;

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
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx >= width * height)
        return;

    float3 input_val = float3(
        input_image_array[3 * idx + 0],
        input_image_array[3 * idx + 1],
        input_image_array[3 * idx + 2]
    );

    float3 rfeat_val = float3(
        rfeat_array[t2i(0, idx, height * width)],
        rfeat_array[t2i(1, idx, height * width)],
        rfeat_array[t2i(2, idx, height * width)]
    );

    float3 diff = rfeat_val - input_val;
    
    // // 计算 PSNR 的部分值 (平方误差之和)
    psnr_sum.InterlockedAddF32(0, diff.x * diff.x);
    psnr_sum.InterlockedAddF32(sizeof(float32_t), diff.y * diff.y);
    psnr_sum.InterlockedAddF32(sizeof(float32_t) * 2, diff.z * diff.z);
}
