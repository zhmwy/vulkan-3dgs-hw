# include "utils.slang"

#define BX 16
#define BY 16

[[vk::binding(0)]] StructuredBuffer<uint8_t> input_image_array;
[[vk::binding(1)]] RWStructuredBuffer<float> alpha_array;
[[vk::binding(2)]] RWStructuredBuffer<float> dL_dalpha_array;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint xyz_size;
    uint width;
    uint height;
    uint temp;
};



[shader("compute")]
[numthreads(BX, BY, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint3 groupID = dispatchThreadID / uint3(BX, BY, 1);
    uint3 groupThreadID = dispatchThreadID % uint3(BX, BY, 1);
    int pix_y = groupID.y * BY + groupThreadID.y;
    int pix_x = groupID.x * BX + groupThreadID.x;

    if (pix_x >= width || pix_y >= height)
        return;

    float gt = 0.0;
    if (input_image_array[t2i(pix_y, pix_x, width)] > 0){
        gt = 1.0;
    }

    float alpha = alpha_array[t2i(pix_y, pix_x, width)];

    if (alpha < 1e-6 || alpha > 1 - 1e-6){
        dL_dalpha_array[t2i(pix_y, pix_x, width)] = 0.0;
    }
    else{
        float ratio = -lamda_mask / height / width;
        dL_dalpha_array[t2i(pix_y, pix_x, width)] = ratio * (gt / alpha -   (1 - gt) / (1 - alpha));
    }
}
