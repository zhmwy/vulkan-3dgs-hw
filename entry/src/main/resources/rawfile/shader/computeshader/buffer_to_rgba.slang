#include "utils.slang"
[[vk::binding(0)]] StructuredBuffer<float> inputBuffer;
[[vk::binding(1)]] RWStructuredBuffer<uint8_t4> outputBuffer;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint xyz_size;
    uint width;
    uint height;
    uint temp;
};


[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    int2 fragCoord = threadId.xy;

    if (fragCoord.x >= width || fragCoord.y >= height)
    {
        return;
    }

    float3 curr_pix = float3(
        inputBuffer[t2i(0, fragCoord.y, fragCoord.x, height, width)],
        inputBuffer[t2i(1, fragCoord.y, fragCoord.x, height, width)],
        inputBuffer[t2i(2, fragCoord.y, fragCoord.x, height, width)]);

    curr_pix = clamp(curr_pix, 0.0, 1.0);

    outputBuffer[t2i(fragCoord.y, fragCoord.x, width)] = uint8_t4(
        uint8_t(curr_pix.x * 255.0 + 0.5),
        uint8_t(curr_pix.y * 255.0 + 0.5),
        uint8_t(curr_pix.z * 255.0 + 0.5), 255);
}
