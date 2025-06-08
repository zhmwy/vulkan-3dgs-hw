# include "utils.slang"
[[vk::binding(0)]] StructuredBuffer<float> input_image_array;
[[vk::binding(1)]] StructuredBuffer<float> rfeat_array;
[[vk::binding(2)]] RWStructuredBuffer<float> dL_drfeat_array;
[[vk::binding(3)]] RWByteAddressBuffer psnr_sum;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint xyz_size;
    uint width;
    uint height;
    uint temp;
};

float3 sign3(float3 v)
{
    float3 res;
    if (v.x > 0.0) res.x = 1.0;
    else res.x = -1.0;
    if (v.y > 0.0) res.y = 1.0;
    else res.y = -1.0;
    if (v.z > 0.0) res.z = 1.0;
    else res.z = -1.0;
    return res;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx >= width * height)
        return;

    // float3 input_val = float3(
    //     input_image_array[3 * idx + 0],
    //     input_image_array[3 * idx + 1],
    //     input_image_array[3 * idx + 2]
    // );

    float3 input_val = float3(
        input_image_array[t2i(0, idx, height * width)],
        input_image_array[t2i(1, idx, height * width)],
        input_image_array[t2i(2, idx, height * width)]
    );

    float3 rfeat_val = float3(
        rfeat_array[t2i(0, idx, height * width)],
        rfeat_array[t2i(1, idx, height * width)],
        rfeat_array[t2i(2, idx, height * width)]
    );

    float3 diff = rfeat_val - input_val;

    // l1 loss
    float3 dL_drfeat = sign3(diff) / (3 * height * width);
    
    // l2 loss
    // float3 dL_drfeat = 2.0 * diff / (3 * height * width);

    dL_drfeat = dL_drfeat * (1.0 - lambda_dssim);     // L1 weight

    dL_drfeat_array[t2i(0, idx, height * width)] += dL_drfeat.x;
    dL_drfeat_array[t2i(1, idx, height * width)] += dL_drfeat.y;
    dL_drfeat_array[t2i(2, idx, height * width)] += dL_drfeat.z;

}
