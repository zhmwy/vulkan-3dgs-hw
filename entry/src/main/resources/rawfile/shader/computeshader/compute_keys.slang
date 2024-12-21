#include "utils.slang"
[CudaKernel]
void computeKeysKernel(
    // inputs-buffers
    TensorView<float3> uvd_array,
    TensorView<int32_t> tiles_array,
    TensorView<int32_t4> tiles_rect_array,
    TensorView<bool> visible_array,
    // input-scalars
    int2 grid,
    // outputs
    TensorView<int64_t> unsorted_keys_array,
    TensorView<int32_t> unsorted_idx_array,
    TensorView<int32_t> num
);

// #extension GL_ARB_gpu_shader_int64 : require

// Buffer bindings
[[vk::binding(0)]] StructuredBuffer<float> uvd_array;
[[vk::binding(1)]] StructuredBuffer<uint32_t> tiles_array;
[[vk::binding(2)]] StructuredBuffer<uint32_t4> tiles_rect_array;
[[vk::binding(3)]] StructuredBuffer<uint32_t> visible_array;

[[vk::binding(4)]] RWStructuredBuffer<uint32_t> unsorted_keys_array;
[[vk::binding(5)]] RWStructuredBuffer<uint32_t> unsorted_idx_array;
[[vk::binding(6)]] RWByteAddressBuffer num;
// [[vk::binding(6)]] RWStructuredBuffer<int32_t> num;
// [[vk::binding(7)]] StructuredBuffer<int32_t> curPointsNum;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint xyz_size;
    uint W;
    uint H;
    uint temp;
};

uint float32ToFloat20(float input) {
    // Get the raw bits of the float
    uint32_t floatBits = reinterpret<uint32_t>(input);

    // Extract the exponent (8 bits) and adjust the bias
    int exponent = int((floatBits >> 23u) & 0xFFu) - 127; // IEEE 754 bias is 127
    exponent = clamp(exponent + 31, 0, 63);               // Adjust to 6-bit bias (31)

    // Extract the mantissa (23 bits) and truncate to 13 bits
    uint mantissa = (floatBits & 0x7FFFFFu) >> 10u; // Keep top 13 bits of mantissa

    // Combine exponent and mantissa into 20-bit representation
    uint float20Bits = (uint(exponent) << 13u) | mantissa;

    return float20Bits;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint32_t idx = dispatchThreadID.x;
    if (idx >= xyz_size)
        return;

    int32_t offset;
    num.InterlockedAdd(0, tiles_array[idx], offset);
    // num.InterlockedAdd(0, 1, offset);

    float3 uvd_reg = float3(
        uvd_array[idx * 3 + 0],
        uvd_array[idx * 3 + 1],
        uvd_array[idx * 3 + 2]);
    int32_t4 tile_rect_reg = tiles_rect_array[idx];
    int32_t rect_min_x = tile_rect_reg[0];
    int32_t rect_min_y = tile_rect_reg[1];
    int32_t rect_max_x = tile_rect_reg[2];
    int32_t rect_max_y = tile_rect_reg[3];
    int grid_width = (W + BLOCK_X - 1) / BLOCK_X;

    for (int32_t y = rect_min_y; y < rect_max_y; y++) {
        for (int32_t x = rect_min_x; x < rect_max_x; x++) {
            uint32_t key = y * grid_width + x;
            key <<= 19;
            uint32_t bits = float32ToFloat20(uvd_reg.z);
            key |= bits;
            unsorted_keys_array[offset] = key;
            unsorted_idx_array[offset] = idx;
            
            offset = offset + 1;
        }
    }
}
