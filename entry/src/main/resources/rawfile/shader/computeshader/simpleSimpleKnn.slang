#include "utils.slang"

[[vk::binding(0)]] StructuredBuffer<float> point_positions;  // 每个点的3D位置，线性展开的float数组
[[vk::binding(1)]] RWStructuredBuffer<float> avg_distances;    // 每个点的最近三个点平均距离

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
    uint idx = dispatchThreadID.x;
    const float LARGE_FLOAT = 1.0e+30f;
    if (idx < 0 || idx >= xyz_size)
        return;

    // 当前点的3D坐标
    float3 currentPoint = float3(
        point_positions[3 * idx],
        point_positions[3 * idx + 1],
        point_positions[3 * idx + 2]
    );

    // 初始化最近距离为一个大值
    float3 nearestDistances = float3(LARGE_FLOAT, LARGE_FLOAT, LARGE_FLOAT);

    // 遍历其他点，计算距离
    for (uint i = 0; i < xyz_size; ++i)
    {
        if (i == idx) // 跳过自己
            continue;

        float3 otherPoint = float3(
            point_positions[3 * i],
            point_positions[3 * i + 1],
            point_positions[3 * i + 2]
        );

        float3 d = float3(
            otherPoint.x - currentPoint.x,
            otherPoint.y - currentPoint.y,
            otherPoint.z - currentPoint.z
        );
        float dist = d.x * d.x + d.y * d.y + d.z * d.z;

        if (dist < EPS) continue;

        // 更新最近的三个距离
        if (nearestDistances.x > dist)
        {
            nearestDistances.z = nearestDistances.y;
            nearestDistances.y = nearestDistances.x;
            nearestDistances.x = dist;
        }
        else if (nearestDistances.y > dist)
        {
            nearestDistances.z = nearestDistances.y;
            nearestDistances.y = dist;
        }
        else if (nearestDistances.z > dist)
        {
            nearestDistances.z = dist;
        }
    }

    // 计算平均值
    // float avgDistance = (nearestDistances.x + nearestDistances.y + nearestDistances.z) / 3.0f;
    float avgDistance = nearestDistances.x;

    // 应用对数变换和结果输出
    float logAvgDistance = log(avgDistance) / 2.0f;
    avg_distances[idx * 3] = logAvgDistance;
    avg_distances[idx * 3 + 1] = logAvgDistance;
    avg_distances[idx * 3 + 2] = logAvgDistance;
}