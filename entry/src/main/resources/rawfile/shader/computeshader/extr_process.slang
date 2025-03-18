[[vk::binding(0)]] StructuredBuffer<float> camera_extrs;
[[vk::binding(1)]] StructuredBuffer<float> xyz_array;
[[vk::binding(2)]] StructuredBuffer<float> extr_params;

[[vk::binding(3)]] StructuredBuffer<uint32_t> visible_array;
[[vk::binding(4)]] RWStructuredBuffer<float> value_array;

[[vk::binding(9)]] StructuredBuffer<int32_t> curPointsNum;

[[vk::binding(10)]] RWStructuredBuffer<float> extr;
[[vk::binding(12)]] RWStructuredBuffer<float> extr_act;
[[vk::binding(11)]] RWStructuredBuffer<float> camera_center;


[[vk::push_constant]]
cbuffer PushConstants
{
    float xyz_size;
    float width;
    float height;
    float position_lr;
    float D;
};
