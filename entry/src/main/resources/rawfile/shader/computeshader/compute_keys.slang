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
[[vk::binding(7)]] StructuredBuffer<int32_t> curPointsNum;

[[vk::binding(8)]] StructuredBuffer<float> conic;
[[vk::binding(9)]] StructuredBuffer<float> opacity_act;

[[vk::binding(10)]] StructuredBuffer<float> opacity_mask_act;



[[vk::push_constant]]
cbuffer PushConstants
{
    uint xyz_size;
    uint W;
    uint H;
    uint temp;
};

inline float evaluate_opacity_factor(const float dx, const float dy, const float3 conic) {
    return 0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
}

float max_contrib_power_rect_gaussian_float(
    float3 conic,
    float3 uvd,
    float2 rect_min,
    float2 rect_max
)
{
    const float x_min_diff = rect_min.x - uvd.x;
    float x_left = 0;
    if (x_min_diff > 0.0f) x_left = 1.0;
    // const float x_left = x_min_diff > 0.0f;

    float not_in_x_range = x_left;
    if (uvd.x > rect_max.x) not_in_x_range += 1.0f;
    // const float not_in_x_range = x_left + (uvd.x > rect_max.x);


    const float y_min_diff = rect_min.y - uvd.y;
    float y_above = 0;
    if (y_min_diff > 0.0f) y_above = 1.0f;
    // const float y_above = y_min_diff > 0.0f;

    float not_in_y_range = y_above;
    if (uvd.y > rect_max.y) not_in_y_range += 1;
    // const float not_in_y_range = y_above + (uvd.y > rect_max.y);

    float max_contrib_power = 0.0f;
    float2 max_pos = { uvd.x, uvd.y };

    if (not_in_y_range + not_in_x_range > 0.0) {
        const float px = x_left * rect_min.x + (1.0f - x_left) * rect_max.x;
        const float py = y_above * rect_min.y + (1.0f - y_above) * rect_max.y;

        const float dx = copysign(float(BLOCK_X - 1), x_min_diff);
        const float dy = copysign(float(BLOCK_Y - 1), y_min_diff);

        const float diffx = uvd.x - px;
        const float diffy = uvd.y - py;

        const float rcp_dxdxcox = 1.0f / ((BLOCK_X - 1) * (BLOCK_X - 1) * conic.x); // = 1.0 / (dx*dx*co.x)
        const float rcp_dydycoz = 1.0f / ((BLOCK_Y - 1) * (BLOCK_Y - 1) * conic.z); // = 1.0 / (dy*dy*co.z)

        const float tx = not_in_y_range * clamp((dx * conic.x * diffx + dx * conic.y * diffy) * rcp_dxdxcox, 0.0, 1.0);
        const float ty = not_in_x_range * clamp((dy * conic.y * diffx + dy * conic.z * diffy) * rcp_dydycoz, 0.0, 1.0);
        max_pos = { px + tx * dx, py + ty * dy };

        const float2 max_pos_diff = { uvd.x - max_pos.x, uvd.y - max_pos.y };
        max_contrib_power = evaluate_opacity_factor(max_pos_diff.x, max_pos_diff.y, conic);
    }

    return max_contrib_power;
}

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
    if (idx.x >= curPointsNum[0] || visible_array[idx] == 0)
        return;

    int32_t offset = 0;
    num.InterlockedAdd(0, tiles_array[idx], offset);
    int32_t offset_to = offset + tiles_array[idx];

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

    const float opacity_factor_threshold = log(opacity_act[idx] * 255.0f);
    const float opacity_factor_threshold2 = log(opacity_mask_act[idx] * 255.0f);
    
    float3 conic_reg = {
        conic[t2i(idx, 0, 3)],
        conic[t2i(idx, 1, 3)],
        conic[t2i(idx, 2, 3)]
    };

    for (int32_t y = rect_min_y; y < rect_max_y; y++) {
        for (int32_t x = rect_min_x; x < rect_max_x; x++) {

            float2 tile_min = { x * BLOCK_X, y * BLOCK_Y };
            float2 tile_max = { (x + 1) * BLOCK_X - 1, (y + 1) * BLOCK_Y - 1 };


            float max_opac_factor = max_contrib_power_rect_gaussian_float(conic_reg, uvd_reg, tile_min, tile_max);

            uint32_t key = y * grid_width + x;
            key <<= 19;
            uint32_t bits = float32ToFloat20(uvd_reg.z);
            key |= bits;

            if (max_opac_factor <= opacity_factor_threshold || max_opac_factor <= opacity_factor_threshold2) {
                unsorted_keys_array[offset] = key;
                unsorted_idx_array[offset] = idx;
                offset++;
            }
        }
    }

    for (; offset < offset_to; offset++) {
        uint32_t key = 10000000;
        key <<= 19;
        const float depth = 10000;
        key = key | float32ToFloat20(depth);
        unsorted_keys_array[offset] = key;
        unsorted_idx_array[offset] = (uint32_t)-1;
    }

}
