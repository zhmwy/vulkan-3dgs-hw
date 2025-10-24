[[vk::binding(0)]] RWStructuredBuffer<float> opacity_array;
[[vk::binding(1)]] StructuredBuffer<int32_t> curPointsNum;

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
    int32_t idx = dispatchThreadID.x;
    if (idx >= curPointsNum[0])
        return;

    float opacity_act = 1 / (1 + exp(-opacity_array[idx]));
    opacity_act = min(opacity_act, 0.01);
    opacity_array[idx] = log(opacity_act / (1 - opacity_act));
}



