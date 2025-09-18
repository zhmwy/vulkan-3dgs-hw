#include "utils.slang"

float4 rotmatrix_to_unit_quaternion(float3x3 R)
{
    float4 quaternion;
    float trace = R[0][0] + R[1][1] + R[2][2];

    if (trace > 0) {
        float S = sqrt(trace + 1.0) * 2.0;
        quaternion[0] = 0.25 * S; // w
        quaternion[1] = (R[2][1] - R[1][2]) / S; // x
        quaternion[2] = (R[0][2] - R[2][0]) / S; // y
        quaternion[3] = (R[1][0] - R[0][1]) / S; // z
    }
    else if ((R[0][0] > R[1][1]) && (R[0][0] > R[2][2])) {
        float S = sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2.0;
        quaternion[0] = (R[2][1] - R[1][2]) / S; // w
        quaternion[1] = 0.25 * S; // x
        quaternion[2] = (R[0][1] + R[1][0]) / S; // y
        quaternion[3] = (R[0][2] + R[2][0]) / S; // z
    } 
    else if (R[1][1] > R[2][2]) {
        float S = sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2.0;
        quaternion[0] = (R[0][2] - R[2][0]) / S; // w
        quaternion[1] = (R[0][1] + R[1][0]) / S; // x
        quaternion[2] = 0.25 * S; // y
        quaternion[3] = (R[1][2] + R[2][1]) / S; // z
    } 
    else {
        float S = sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2.0;
        quaternion[0] = (R[1][0] - R[0][1]) / S; // w
        quaternion[1] = (R[0][2] + R[2][0]) / S; // x
        quaternion[2] = (R[1][2] + R[2][1]) / S; // y
        quaternion[3] = 0.25 * S; // z
    }

    return quaternion;
}


[[vk::binding(0)]] StructuredBuffer<float> point_positions;  // 每个点的3D位置，线性展开的float数组
[[vk::binding(1)]] RWStructuredBuffer<float> avg_distances;    // 每个点的最近三个点平均距离
[[vk::binding(2)]] StructuredBuffer<float> normals;  // 每个点的3D位置，线性展开的float数组
[[vk::binding(3)]] RWStructuredBuffer<float> rotation_array;    // 每个点的最近三个点平均距离


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
    return;






















    
    if (idx < skybox_num)
        return;

    float3 normal = float3(
        normals[3 * idx],
        normals[3 * idx + 1],
        normals[3 * idx + 2]
    );

    float d = sqrt(normal.x * normal.x + normal.y * normal.y);

    float3x3 R;
    if (normal.z == 1) {
        R = float3x3(
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        );
    }

    else if (normal.z == -1) {
        R = float3x3(
            1, 0, 0,
            0, -1, 0,
            0, 0, -1
        );
    }

    else {
        R = float3x3(
            -normal.y / d, -normal.x * normal.z / d, normal.x,
             normal.x / d, -normal.y * normal.z / d, normal.y,
            0,              d,                       normal.z
        );
    }

    // avg_distances[idx * 3 + 0] *= 1.1;
    // avg_distances[idx * 3 + 1] *= 1.1;
    // avg_distances[idx * 3 + 2] /= 2;

    // float4 q = rotmatrix_to_unit_quaternion(R);
    // rotation_array[idx *4 + 0] = q[0];
    // rotation_array[idx *4 + 1] = q[1];
    // rotation_array[idx *4 + 2] = q[2];
    // rotation_array[idx *4 + 3] = q[3];
    
}