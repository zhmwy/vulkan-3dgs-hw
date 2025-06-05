#include "utils.slang"

float3x3 scale_vector_to_matrix(float3 scale) {
    float3x3 S = float3x3(
        scale.x, 0, 0,
        0, scale.y, 0,
        0, 0, scale.z
    );
    return S;
}

float3x3 unit_quaternion_to_rotmatrix(float4 rotate) {
    float w = rotate[0];
    float x = rotate[1];
    float y = rotate[2];
    float z = rotate[3];

    return float3x3(
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)
    );
}

int4 get_rect(
    float3 uvd,
    int max_radius,
    int2 grid
)
{
    int4 range = { 0, 0, 0, 0 };
    range[0] = min(grid.x, max(int(uvd.x - max_radius) / BLOCK_X, 0));
    range[1] = min(grid.y, max(int(uvd.y - max_radius) / BLOCK_Y, 0));
    range[2] = min(grid.x, max(int(uvd.x + max_radius + BLOCK_X - 1) / BLOCK_X, 0));
    range[3] = min(grid.y, max(int(uvd.y + max_radius + BLOCK_Y - 1) / BLOCK_Y, 0));
    return range;
}

// Buffer bindings
[[vk::binding(0)]] StructuredBuffer<float> xyz_array;
[[vk::binding(1)]] StructuredBuffer<float> scale_array;
[[vk::binding(2)]] StructuredBuffer<float> rotate_array;
[[vk::binding(3)]] StructuredBuffer<float> intr_params;
[[vk::binding(4)]] StructuredBuffer<float> extr_params;    // [w, x, y, z, tx, ty, tz]

[[vk::binding(5)]] RWStructuredBuffer<uint32_t> visible_array;
[[vk::binding(6)]] RWStructuredBuffer<float> uvd_array;
[[vk::binding(7)]] RWStructuredBuffer<float> cov2d_array;
[[vk::binding(8)]] RWStructuredBuffer<float> cov3d_array;
[[vk::binding(9)]] RWStructuredBuffer<int32_t> radius;
[[vk::binding(10)]] RWStructuredBuffer<int32_t> tiles_torched;
[[vk::binding(11)]] RWStructuredBuffer<int32_t> tiles_range;

// patch with activation
[[vk::binding(19)]] RWStructuredBuffer<float> opacity_array;
[[vk::binding(20)]] RWStructuredBuffer<float> opacity_act_array;

[[vk::binding(23)]] StructuredBuffer<int32_t> curPointsNum;

[[vk::binding(24)]] RWStructuredBuffer<float> extr;         // 12 parameters
[[vk::binding(25)]] RWStructuredBuffer<float> extr_act;     // [w, x, y, z, tx, ty, tz]  normalized


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
    uint32_t idx = dispatchThreadID.x;
    if (idx.x >= curPointsNum[0])
        return;

    if (visible_array[idx] == 0)
        return;

    float intr[4] = { intr_params[0], intr_params[1], intr_params[2], intr_params[3] };

    // 0. activate opacity, scale and rotate
    float3 scale_reg = {
        scale_array[3 * idx],
        scale_array[3 * idx + 1],
        scale_array[3 * idx + 2]
    };

    float4 rotate_reg = {
        rotate_array[4 * idx],
        rotate_array[4 * idx + 1],
        rotate_array[4 * idx + 2],
        rotate_array[4 * idx + 3]
    };

    float3 scale_act_reg = exp(scale_reg);
    opacity_act_array[idx] = 1 / (1 + exp(-opacity_array[idx]));
    if (opacity_act_array[idx] < min_opacity) {
        visible_array[idx] = 0;
        return;
    }

    float norm_rotate = sqrt(rotate_reg[0] * rotate_reg[0] +
                             rotate_reg[1] * rotate_reg[1] +
                             rotate_reg[2] * rotate_reg[2] +
                             rotate_reg[3] * rotate_reg[3]);
    float4 rotate_act_reg = rotate_reg / norm_rotate;

    // 1. project point
    // float3 xw = xyz_array[idx];
    float3 xw = {
        xyz_array[3 * idx],
        xyz_array[3 * idx + 1],
        xyz_array[3 * idx + 2]
    };
    float3 xc = {
        extr[0] * xw.x + extr[1] * xw.y + extr[2] * xw.z + extr[3],
        extr[4] * xw.x + extr[5] * xw.y + extr[6] * xw.z + extr[7],
        extr[8] * xw.x + extr[9] * xw.y + extr[10] * xw.z + extr[11]
    };

    if (xc.z < 0.02) {
        visible_array[idx] = 0;
        return;
    }

    float norm = 1.0 / xc.z;
    float3 uvd_reg = float3(
        intr[0] * xc.x * norm + intr[2] - 0.5f,
        intr[1] * xc.y * norm + intr[3] - 0.5f,
        xc.z
    );

    float x_limit_min = -0.15 * width;
    float x_limit_max = 1.15 * width;
    float y_limit_min = -0.15 * height;
    float y_limit_max = 1.15 * height;
    if (uvd_reg.x < x_limit_min || uvd_reg.x > x_limit_max ||
        uvd_reg.y < y_limit_min || uvd_reg.y > y_limit_max) {
        visible_array[idx] = 0;
        return;
    }

    uvd_array[3 * idx] = uvd_reg.x;
    uvd_array[3 * idx + 1] = uvd_reg.y;
    uvd_array[3 * idx + 2] = uvd_reg.z;

    // 2. compute cov3d
    float3x3 S = scale_vector_to_matrix(scale_act_reg);
    float3x3 R = unit_quaternion_to_rotmatrix(rotate_act_reg);

    float3x3 RS = mul(R, S);
    float3x3 Sigma = mul(RS, transpose(RS));

    cov3d_array[t2i(idx, 0, 6)] = Sigma[0][0];
    cov3d_array[t2i(idx, 1, 6)] = Sigma[0][1];
    cov3d_array[t2i(idx, 2, 6)] = Sigma[0][2];
    cov3d_array[t2i(idx, 3, 6)] = Sigma[1][1];
    cov3d_array[t2i(idx, 4, 6)] = Sigma[1][2];
    cov3d_array[t2i(idx, 5, 6)] = Sigma[2][2];

    // 3. EWA project
    float3x3 Rw2c = float3x3(
        extr[0], extr[1], extr[2],
        extr[4], extr[5], extr[6],
        extr[8], extr[9], extr[10]
    );
    float2x3 J = float2x3(
        intr[0] / xc.z, 0.0, -(intr[0] * xc.x) / (xc.z * xc.z),
        0.0, intr[1] / xc.z, -(intr[1] * xc.y) / (xc.z * xc.z)
    );

    float2x3 T = mul(J, Rw2c);
    float2x2 cov2d = mul(T, mul(Sigma, transpose(T)));

    float3 cov = float3(
        cov2d[0][0] + 0.3f,
        cov2d[0][1],
        cov2d[1][1] + 0.3f
    );

    float det = cov.x * cov.z - cov.y * cov.y;
    if (det < EPS)
        return;

    cov2d_array[3 * idx] = cov.x;
    cov2d_array[3 * idx + 1] = cov.y;
    cov2d_array[3 * idx + 2] = cov.z;

    // compute radius and tile range.
    float mid = 0.5f * (cov.x + cov.z);
    float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
    int2 grid = int2((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);
    // int2 grid = int2(width, height);
    int4 range = get_rect(uvd_reg, int(my_radius), grid);

    if ((range[2] - range[0]) * (range[3] - range[1]) <= 0)
        return;
    radius[idx] = int32_t(my_radius);
    tiles_torched[idx] = ((range[2] - range[0]) * (range[3] - range[1]));

    tiles_range[4 * idx] = range[0];
    tiles_range[4 * idx + 1] = range[1];
    tiles_range[4 * idx + 2] = range[2];
    tiles_range[4 * idx + 3] = range[3];
}
