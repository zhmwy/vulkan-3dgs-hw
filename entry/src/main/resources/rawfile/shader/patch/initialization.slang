#include "utils.slang"

[[vk::binding(0)]] StructuredBuffer<float> img1;
[[vk::binding(1)]] StructuredBuffer<float> img2;
[[vk::binding(2)]] RWStructuredBuffer<float> dm_dmu1;

