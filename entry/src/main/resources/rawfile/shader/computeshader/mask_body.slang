#include "utils.slang"

[[vk::binding(0)]] StructuredBuffer<float> xyz_array;
[[vk::binding(1)]] RWStructuredBuffer<int32_t> is_body_point;
[[vk::binding(2)]] StructuredBuffer<float> body_bbox;
[[vk::binding(3)]] StructuredBuffer<int32_t> curPointsNum;
[[vk::binding(4)]] StructuredBuffer<uint32_t> visible_array;

[[vk::push_constant]] //这是一些常数，外面用vkCmdPushConstants
cbuffer PushConstants
{
    uint xyz_size;
    uint width;
    uint height;
    uint temp;
};

[shader("compute")]
[numthreads(256, 1, 1)] //这个跟C++里写的得一样，外面除以256，这里第一个数是256
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID) //主函数都是computeMain这个名字，参数也是一样的
{
    uint32_t idx = dispatchThreadID.x; //idx就对应第几个xyz
    if (idx >= curPointsNum[0] || visible_array[idx] == 0){
        is_body_point[idx] = 0;
        return;
    }
    float x = xyz_array[3 * idx];
    float y = xyz_array[3 * idx + 1];
    float z = xyz_array[3 * idx + 2];
    float x_min = body_bbox[0];
    float x_max = body_bbox[1];
    float y_min = body_bbox[2];
    float y_max = body_bbox[3];
    float z_min = body_bbox[4];
    float z_max = body_bbox[5];

    if(x > x_min && x < x_max && y > y_min && y < y_max && z > z_min && z < z_max){
        is_body_point[idx] = 1;
    }
    else{
        is_body_point[idx] = 0;
    }
}
