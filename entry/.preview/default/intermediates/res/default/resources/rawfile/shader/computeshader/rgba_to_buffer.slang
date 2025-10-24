// hello-world.slang
[[vk::binding(0)]] RWStructuredBuffer<uint8_t4> inputBuffer;
[[vk::binding(1)]] RWStructuredBuffer<float> outputBuffer;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint xyz_size;
    uint width;
    uint height;
    uint temp;
};

uint t2i(uint a, uint b, uint x) // tuple to index
{
    return a * x + b;
}

uint t2i(uint c, uint i, uint j, uint height, uint width) {
    return c * height * width + i * width + j;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    int2 fragCoord = threadId.xy;

    if (fragCoord.x >= width || fragCoord.y >= height)
    {
        return;
    }

    float4 curr_pix = inputBuffer[t2i(fragCoord.y, fragCoord.x, width)];

    uint idx = t2i(fragCoord.y, fragCoord.x, width);

    // outputBuffer[idx * 3 + 0] = curr_pix.x / 255.0;
    // outputBuffer[idx * 3 + 1] = curr_pix.y / 255.0;
    // outputBuffer[idx * 3 + 2] = curr_pix.z / 255.0;

    outputBuffer[t2i(0, fragCoord.y, fragCoord.x, height, width)] = curr_pix.x / 255.0;
    outputBuffer[t2i(1, fragCoord.y, fragCoord.x, height, width)] = curr_pix.y / 255.0;
    outputBuffer[t2i(2, fragCoord.y, fragCoord.x, height, width)] = curr_pix.z / 255.0;
}
