#include "utils.slang"

[[vk::binding(0)]] StructuredBuffer<float> images;
[[vk::binding(1)]] StructuredBuffer<int>   srcPairs;
[[vk::binding(2)]] StructuredBuffer<float> cameraIntrics;
[[vk::binding(3)]] StructuredBuffer<float> depthRanges;
[[vk::binding(4)]] RWStructuredBuffer<float4> planeHypotheses;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint width;
    uint height;
    uint srcCount;

    uint currentImage;
};

float4 GenerateRandomPlaneHypothesis(float4 cameraK, const int2 p, const float depth_min, const float depth_max)
{
    float depth = random(p, 1) * (depth_max - depth_min) + depth_min;
    float4 plane_hypothesis = GenerateRandomNormal(cameraK, p, depth);
    plane_hypothesis.w = GetDistance2Origin(cameraK, p, depth, plane_hypothesis);

    return plane_hypothesis;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint2 p = {
        dispatchThreadID.x,
        dispatchThreadID.y
    };

    if (p.x >= width || p.y >= height) {
        return ;
    }

    float4 cameraK = {
        cameraIntrics[currentImage*4+0],
        cameraIntrics[currentImage*4+1],
        cameraIntrics[currentImage*4+2],
        cameraIntrics[currentImage*4+3]
    };

    float depth_min = depthRanges[currentImage*2+0];
    float depth_max = depthRanges[currentImage*2+1];

    planeHypotheses[p.y * width + p.x] = GenerateRandomPlaneHypothesis(cameraK, p, depth_min, depth_max);
    planeHypotheses[p.y * width + p.x] = {1.0, 0.0, 0.0, 1.0};
}



