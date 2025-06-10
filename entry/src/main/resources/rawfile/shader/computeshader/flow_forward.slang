#include "utils.slang"

// Buffer bindings
// 这里是26个buffer，StructuredBuffer是只读的，RWStructuredBuffer是可以写的
// 原子加是为了多个线程对同一个变量做加法，为了避免线程冲突，必须原子加（比如点数，在densify的时候）；
[[vk::binding(0)]] StructuredBuffer<float> time_stamp;
[[vk::binding(1)]] RWStructuredBuffer<float> xyz_array;
[[vk::binding(2)]] RWStructuredBuffer<float> rotate_array;
[[vk::binding(3)]] RWStructuredBuffer<float> xyz_flow;
[[vk::binding(4)]] RWStructuredBuffer<float> rot_flow;
[[vk::binding(5)]] StructuredBuffer<float> xyz_flow_params;
[[vk::binding(6)]] StructuredBuffer<float> rot_flow_params;
[[vk::binding(7)]] StructuredBuffer<int32_t> curPointsNum;
[[vk::binding(8)]] StructuredBuffer<uint32_t> visible_array;
[[vk::binding(13)]] StructuredBuffer<int32_t> is_body_point;
[[vk::binding(14)]] StructuredBuffer<float> max_movement;

[[vk::push_constant]] //这是一些常数，外面用vkCmdPushConstants
cbuffer PushConstants
{
    uint xyz_size;
    uint width;
    uint height;
    uint temp;
};

void xyz_traj_func(float timestamp, float3x3 an, float3x3 fl, int idx){
    float x_flow = an[0][0] * timestamp + an[0][1] * pow(timestamp, 2) + an[0][2] * pow(timestamp, 3) + fl[0][0] * sin(2 * pi * timestamp) + fl[0][1] * sin(2 * pi * 2 * timestamp) + fl[0][2] * sin(2 * pi * 3 * timestamp);
    float y_flow = an[1][0] * timestamp + an[1][1] * pow(timestamp, 2) + an[1][2] * pow(timestamp, 3) + fl[1][0] * sin(2 * pi * timestamp) + fl[1][1] * sin(2 * pi * 2 * timestamp) + fl[1][2] * sin(2 * pi * 3 * timestamp);
    float z_flow = an[2][0] * timestamp + an[2][1] * pow(timestamp, 2) + an[2][2] * pow(timestamp, 3) + fl[2][0] * sin(2 * pi * timestamp) + fl[2][1] * sin(2 * pi * 2 * timestamp) + fl[2][2] * sin(2 * pi * 3 * timestamp);

    // float max_flow = max3<float>(abs(x_flow), abs(y_flow), abs(z_flow));
    // float scale = 1.0 / max<float>(1.0, max_flow / max_movement[0]);
    // flow_scalings[idx] = scale;
    // float scale = 1.0;

    xyz_array[3 * idx] += x_flow / 3.0;
    xyz_array[3 * idx + 1] += y_flow / 3.0;
    xyz_array[3 * idx + 2] += z_flow / 3.0;
    xyz_flow[3 * idx] = x_flow / 3.0;
    xyz_flow[3 * idx + 1] = y_flow / 3.0;
    xyz_flow[3 * idx + 2] = z_flow / 3.0;
}

void rot_traj_func(float timestamp, float4x3 an, float4x3 fl, int idx){
    float qw_flow = an[0][0] * timestamp + an[0][1] * pow(timestamp, 2) + an[0][2] * pow(timestamp, 3) + fl[0][0] * sin(2 * pi * timestamp) + fl[0][1] * sin(2 * pi * 2 * timestamp) + fl[0][2] * sin(2 * pi * 3 * timestamp);
    float qx_flow = an[1][0] * timestamp + an[1][1] * pow(timestamp, 2) + an[1][2] * pow(timestamp, 3) + fl[1][0] * sin(2 * pi * timestamp) + fl[1][1] * sin(2 * pi * 2 * timestamp) + fl[1][2] * sin(2 * pi * 3 * timestamp);
    float qy_flow = an[2][0] * timestamp + an[2][1] * pow(timestamp, 2) + an[2][2] * pow(timestamp, 3) + fl[2][0] * sin(2 * pi * timestamp) + fl[2][1] * sin(2 * pi * 2 * timestamp) + fl[2][2] * sin(2 * pi * 3 * timestamp);
    float qz_flow = an[3][0] * timestamp + an[3][1] * pow(timestamp, 2) + an[3][2] * pow(timestamp, 3) + fl[3][0] * sin(2 * pi * timestamp) + fl[3][1] * sin(2 * pi * 2 * timestamp) + fl[3][2] * sin(2 * pi * 3 * timestamp);

    // rotate_array[4 * idx] += (an[0][0] * timestamp + an[0][1] * pow(timestamp, 2) + an[0][2] * pow(timestamp, 3))/3;  // qw
    // rotate_array[4 * idx + 1] += (an[1][0] * timestamp + an[1][1] * pow(timestamp, 2) + an[1][2] * pow(timestamp, 3))/3;  // qx
    // rotate_array[4 * idx + 2] += (an[2][0] * timestamp + an[2][1] * pow(timestamp, 2) + an[2][2] * pow(timestamp, 3))/3;  // qy
    // rotate_array[4 * idx + 3] += (an[3][0] * timestamp + an[3][1] * pow(timestamp, 2) + an[3][2] * pow(timestamp, 3))/3;  // qz

    // rotate_array[4 * idx] += (fl[0][0] * sin(2 * pi * timestamp) + fl[0][1] * sin(2 * pi * 2 * timestamp) + fl[0][2] * sin(2 * pi * 3 * timestamp))/3;  // qw
    // rotate_array[4 * idx + 1] += (fl[1][0] * sin(2 * pi * timestamp) + fl[1][1] * sin(2 * pi * 2 * timestamp) + fl[1][2] * sin(2 * pi * 3 * timestamp))/3;  // qx
    // rotate_array[4 * idx + 2] += (fl[2][0] * sin(2 * pi * timestamp) + fl[2][1] * sin(2 * pi * 2 * timestamp) + fl[2][2] * sin(2 * pi * 3 *timestamp))/3;  // qy
    // rotate_array[4 * idx + 3] += (fl[3][0] * sin(2 * pi * timestamp) + fl[3][1] * sin(2 * pi * 2 * timestamp) + fl[3][2] * sin(2 * pi * 3 *timestamp))/3;  // qz
    rotate_array[4 * idx] += qw_flow / 3.0;
    rotate_array[4 * idx + 1] += qx_flow / 3.0;
    rotate_array[4 * idx + 2] += qy_flow / 3.0;
    rotate_array[4 * idx + 3] += qz_flow / 3.0;
    rot_flow[4 * idx] = qw_flow / 3.0;
    rot_flow[4 * idx + 1] = qx_flow / 3.0;
    rot_flow[4 * idx + 2] = qy_flow / 3.0;
    rot_flow[4 * idx + 3] = qz_flow / 3.0;
}

[shader("compute")]
[numthreads(256, 1, 1)] //这个跟C++里写的得一样，外面除以256，这里第一个数是256
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID) //主函数都是computeMain这个名字，参数也是一样的
{
    uint32_t idx = dispatchThreadID.x; //idx就对应第几个xyz
    if (idx >= curPointsNum[0] || visible_array[idx] == 0 || is_body_point[idx] == 0)
        return;
        
    float timestamp = time_stamp[0];
    
    float3x3 an_xyz = {
        xyz_flow_params[18 * idx], xyz_flow_params[18 * idx + 1], xyz_flow_params[18 * idx + 2],
        xyz_flow_params[18 * idx + 3], xyz_flow_params[18 * idx + 4], xyz_flow_params[18 * idx + 5],
        xyz_flow_params[18 * idx + 6], xyz_flow_params[18 * idx + 7], xyz_flow_params[18 * idx + 8]
    };

    float3x3 fl_xyz = {
        xyz_flow_params[18 * idx + 9], xyz_flow_params[18 * idx + 10], xyz_flow_params[18 * idx + 11],
        xyz_flow_params[18 * idx + 12], xyz_flow_params[18 * idx + 13], xyz_flow_params[18 * idx + 14],
        xyz_flow_params[18 * idx + 15], xyz_flow_params[18 * idx + 16], xyz_flow_params[18 * idx + 17]
    };

    float4x3 an_rot = {
        rot_flow_params[24 * idx], rot_flow_params[24 * idx + 1], rot_flow_params[24 * idx + 2],
        rot_flow_params[24 * idx + 3], rot_flow_params[24 * idx + 4], rot_flow_params[24 * idx + 5],
        rot_flow_params[24 * idx + 6], rot_flow_params[24 * idx + 7], rot_flow_params[24 * idx + 8],
        rot_flow_params[24 * idx + 9], rot_flow_params[24 * idx + 10], rot_flow_params[24 * idx + 11]
    };

    float4x3 fl_rot = {
        rot_flow_params[24 * idx + 12], rot_flow_params[24 * idx + 13], rot_flow_params[24 * idx + 14],
        rot_flow_params[24 * idx + 15], rot_flow_params[24 * idx + 16], rot_flow_params[24 * idx + 17],
        rot_flow_params[24 * idx + 18], rot_flow_params[24 * idx + 19], rot_flow_params[24 * idx + 20],
        rot_flow_params[24 * idx + 21], rot_flow_params[24 * idx + 22], rot_flow_params[24 * idx + 23]
    };

    xyz_traj_func(timestamp, an_xyz, fl_xyz, idx);
    rot_traj_func(timestamp, an_rot, fl_rot, idx);
}
