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
[[vk::binding(9)]] StructuredBuffer<float> dL_dxyz;
[[vk::binding(10)]] StructuredBuffer<float> dL_drotate;
[[vk::binding(11)]] RWStructuredBuffer<float> dL_dxyz_flow;
[[vk::binding(12)]] RWStructuredBuffer<float> dL_drot_flow;
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

void calculate_xyz_gradients(int idx){
    float timestamp = time_stamp[0];
    float dL_dx = (1.0 / 3.0) * dL_dxyz[3 * idx];
    float dL_dy = (1.0 / 3.0) * dL_dxyz[3 * idx + 1];
    float dL_dz = (1.0 / 3.0) * dL_dxyz[3 * idx + 2];
    
    // x
    dL_dxyz_flow[18 * idx] = dL_dx * timestamp; //a0
    dL_dxyz_flow[18 * idx + 1] = dL_dx * pow(timestamp, 2); //a1
    dL_dxyz_flow[18 * idx + 2] = dL_dx * pow(timestamp, 3); //a2
    dL_dxyz_flow[18 * idx + 9] = dL_dx * sin(2 * pi * timestamp); //f0
    dL_dxyz_flow[18 * idx + 10] = dL_dx * sin(2 * pi * 2 * timestamp); //f1
    dL_dxyz_flow[18 * idx + 11] = dL_dx * sin(2 * pi * 3 * timestamp); //f2

    // y
    dL_dxyz_flow[18 * idx + 3] = dL_dy * timestamp; //a0
    dL_dxyz_flow[18 * idx + 4] = dL_dy * pow(timestamp, 2); //a1
    dL_dxyz_flow[18 * idx + 5] = dL_dy * pow(timestamp, 3); //a2
    dL_dxyz_flow[18 * idx + 12] = dL_dy * sin(2 * pi * timestamp); //f0
    dL_dxyz_flow[18 * idx + 13] = dL_dy * sin(2 * pi * 2 * timestamp); //f1
    dL_dxyz_flow[18 * idx + 14] = dL_dy * sin(2 * pi * 3 * timestamp); //f2

    // z
    dL_dxyz_flow[18 * idx + 6] = dL_dz * timestamp; //a0
    dL_dxyz_flow[18 * idx + 7] = dL_dz * pow(timestamp, 2); //a1
    dL_dxyz_flow[18 * idx + 8] = dL_dz * pow(timestamp, 3); //a2
    dL_dxyz_flow[18 * idx + 15] = dL_dz * sin(2 * pi * timestamp); //f0
    dL_dxyz_flow[18 * idx + 16] = dL_dz * sin(2 * pi * 2 * timestamp); //f1
    dL_dxyz_flow[18 * idx + 17] = dL_dz * sin(2 * pi * 3 * timestamp); //f2
}

void calculate_rot_gradients(int idx){
    float timestamp = time_stamp[0];
    float dL_dqw = (1.0/3.0) * dL_drotate[4 * idx];
    float dL_dqx = (1.0/3.0) * dL_drotate[4 * idx + 1];
    float dL_dqy = (1.0/3.0) * dL_drotate[4 * idx + 2];
    float dL_dqz = (1.0/3.0) * dL_drotate[4 * idx + 3];

    // qw
    dL_drot_flow[24 * idx] = dL_dqw * timestamp; //a0
    dL_drot_flow[24 * idx + 1] = dL_dqw * pow(timestamp, 2); //a1
    dL_drot_flow[24 * idx + 2] = dL_dqw * pow(timestamp, 3); //a2
    dL_drot_flow[24 * idx + 12] = dL_dqw * sin(2 * pi * timestamp); //f0
    dL_drot_flow[24 * idx + 13] = dL_dqw * sin(2 * pi * 2 * timestamp); //f1
    dL_drot_flow[24 * idx + 14] = dL_dqw * sin(2 * pi * 3 * timestamp); //f2

    // qx
    dL_drot_flow[24 * idx + 3] = dL_dqx * timestamp; //a0
    dL_drot_flow[24 * idx + 4] = dL_dqx * pow(timestamp, 2); //a1
    dL_drot_flow[24 * idx + 5] = dL_dqx * pow(timestamp, 3); //a2
    dL_drot_flow[24 * idx + 15] = dL_dqx * sin(2 * pi * timestamp); //f0
    dL_drot_flow[24 * idx + 16] = dL_dqx * sin(2 * pi * 2 * timestamp); //f1
    dL_drot_flow[24 * idx + 17] = dL_dqx * sin(2 * pi * 3 * timestamp); //f2

    // qy.
    dL_drot_flow[24 * idx + 6] = dL_dqy * timestamp; //a0
    dL_drot_flow[24 * idx + 7] = dL_dqy * pow(timestamp, 2); //a1
    dL_drot_flow[24 * idx + 8] = dL_dqy * pow(timestamp, 3); //a2
    dL_drot_flow[24 * idx + 18] = dL_dqy * sin(2 * pi * timestamp); //f0
    dL_drot_flow[24 * idx + 19] = dL_dqy * sin(2 * pi * 2 * timestamp); //f1
    dL_drot_flow[24 * idx + 20] = dL_dqy * sin(2 * pi * 3 * timestamp); //f2

    // qz
    dL_drot_flow[24 * idx + 9] = dL_dqz * timestamp; //a0
    dL_drot_flow[24 * idx + 10] = dL_dqz * pow(timestamp, 2); //a1
    dL_drot_flow[24 * idx + 11] = dL_dqz * pow(timestamp, 3); //a2
    dL_drot_flow[24 * idx + 21] = dL_dqz * sin(2 * pi * timestamp); //f0
    dL_drot_flow[24 * idx + 22] = dL_dqz * sin(2 * pi * 2 * timestamp); //f1
    dL_drot_flow[24 * idx + 23] = dL_dqz * sin(2 * pi * 3 * timestamp); //f2
}

bool exceeds_threshold(float flow){
    if(abs(flow) > max_movement[0]){
        return true;
    }
    else{
        return false;
    }
}

void calculate_xyz_regulations(int idx){
    float timestamp = time_stamp[0];
    // x
    float x_flow = xyz_flow[3 * idx];
    if(exceeds_threshold(x_flow)){
        float grad =(abs(x_flow) - max_movement[0]) * (2.0 / 3.0) * sign(x_flow);
        dL_dxyz_flow[18 * idx] += grad * timestamp; //a0
        dL_dxyz_flow[18 * idx + 1] += grad * pow(timestamp, 2); //a1
        dL_dxyz_flow[18 * idx + 2] += grad * pow(timestamp, 3); //a2
        dL_dxyz_flow[18 * idx + 9] += grad * sin(2 * pi * timestamp); //f0
        dL_dxyz_flow[18 * idx + 10] += grad * sin(2 * pi * 2 * timestamp); //f1
        dL_dxyz_flow[18 * idx + 11] += grad * sin(2 * pi * 3 * timestamp); //f2
    }
    
    // y
    float y_flow = xyz_flow[3 * idx + 1];
    if(exceeds_threshold(y_flow)){
        float grad =(abs(y_flow) - max_movement[0]) * (2.0 / 3.0) * sign(y_flow);
        dL_dxyz_flow[18 * idx + 3] += grad * timestamp; //a0
        dL_dxyz_flow[18 * idx + 4] += grad * pow(timestamp, 2); //a1
        dL_dxyz_flow[18 * idx + 5] += grad * pow(timestamp, 3); //a2
        dL_dxyz_flow[18 * idx + 12] += grad * sin(2 * pi * timestamp); //f0
        dL_dxyz_flow[18 * idx + 13] += grad * sin(2 * pi * 2 * timestamp); //f1
        dL_dxyz_flow[18 * idx + 14] += grad * sin(2 * pi * 3 * timestamp); //f2
    }

    // z
    float z_flow = xyz_flow[3 * idx + 2];
    if(exceeds_threshold(z_flow)){
        float grad =(abs(z_flow) - max_movement[0]) * (2.0 / 3.0) * sign(z_flow);
        dL_dxyz_flow[18 * idx + 6] += grad * timestamp; //a0
        dL_dxyz_flow[18 * idx + 7] += grad * pow(timestamp, 2); //a1
        dL_dxyz_flow[18 * idx + 8] += grad * pow(timestamp, 3); //a2
        dL_dxyz_flow[18 * idx + 15] += grad * sin(2 * pi * timestamp); //f0
        dL_dxyz_flow[18 * idx + 16] += grad * sin(2 * pi * 2 * timestamp); //f1
        dL_dxyz_flow[18 * idx + 17] += grad * sin(2 * pi * 3 * timestamp); //f2
    }
}

[shader("compute")]
[numthreads(256, 1, 1)] //这个跟C++里写的得一样，外面除以256，这里第一个数是256
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID) //主函数都是computeMain这个名字，参数也是一样的
{
    uint32_t idx = dispatchThreadID.x; //idx就对应第几个xyz
    if (idx >= curPointsNum[0] || visible_array[idx] == 0 || is_body_point[idx] == 0)
        return;
    
    calculate_xyz_gradients(idx);
    calculate_rot_gradients(idx);
    calculate_xyz_regulations(idx);
}