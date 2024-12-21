[[vk::binding(0)]] RWStructuredBuffer<int32_t> InputBuffer;

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
    uint32_t idx = dispatchThreadID.x;
    if (idx >= xyz_size)
        return;

    InputBuffer[idx] = 1;
}
