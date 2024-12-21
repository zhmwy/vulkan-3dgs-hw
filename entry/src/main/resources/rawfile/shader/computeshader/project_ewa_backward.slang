#include "utils.slang"

#extension GL_EXT_shader_atomic_float : require

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

// Buffer bindings
[[vk::binding(0)]] StructuredBuffer<float> xyz_array;
[[vk::binding(1)]] StructuredBuffer<float> scale_array;
[[vk::binding(2)]] StructuredBuffer<float> rotate_array;
[[vk::binding(3)]] StructuredBuffer<float> intr_params;
[[vk::binding(4)]] StructuredBuffer<float> extr_params;
[[vk::binding(5)]] StructuredBuffer<uint32_t> visible_array;
[[vk::binding(8)]] StructuredBuffer<float> cov3d_array;
[[vk::binding(12)]] StructuredBuffer<float> dL_duvd_array;

[[vk::binding(13)]] RWStructuredBuffer<float> dL_dcov2d_array;
[[vk::binding(14)]] RWStructuredBuffer<float> dL_dxyz_array;
[[vk::binding(15)]] RWStructuredBuffer<float> dL_dscale_array;
[[vk::binding(16)]] RWStructuredBuffer<float4> dL_drotate_array;
[[vk::binding(17)]] RWByteAddressBuffer dL_dintr_params;
[[vk::binding(18)]] RWByteAddressBuffer dL_dextr_params;

// patch with activation
[[vk::binding(20)]] RWStructuredBuffer<float> opacity_act_array;
[[vk::binding(21)]] RWStructuredBuffer<float> dL_dopacity_act_array;
[[vk::binding(22)]] RWStructuredBuffer<float> dL_dopacity_array;


[[vk::push_constant]]
cbuffer PushConstants
{
    uint xyz_size;
    uint width;
    uint height;
    uint temp;
};


uint t2i(uint a, uint b) // tuple to index
{
    return a * 4 + b;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint32_t idx = dispatchThreadID.x;
    if (idx.x >= xyz_size || visible_array[idx] == 0) {
        dL_dxyz_array[idx * 3 + 0] = 0;
        dL_dxyz_array[idx * 3 + 1] = 0;
        dL_dxyz_array[idx * 3 + 2] = 0;

        dL_dscale_array[idx * 3 + 0] = 0;
        dL_dscale_array[idx * 3 + 1] = 0;
        dL_dscale_array[idx * 3 + 2] = 0;

        dL_drotate_array[idx * 4 + 0] = 0;
        dL_drotate_array[idx * 4 + 1] = 0;
        dL_drotate_array[idx * 4 + 2] = 0;
        dL_drotate_array[idx * 4 + 3] = 0;

        dL_dopacity_array[idx] = 0;

        return;
    }
        

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
    float norm_rotate = sqrt(rotate_reg[0] * rotate_reg[0] +
                             rotate_reg[1] * rotate_reg[1] +
                             rotate_reg[2] * rotate_reg[2] +
                             rotate_reg[3] * rotate_reg[3]);
    float4 rotate_act_reg = rotate_reg / norm_rotate;

    float intr_reg[4] = { intr_params[0], intr_params[1], intr_params[2], intr_params[3] };
    float extr_reg[12] = {
        extr_params[t2i(0, 0)], extr_params[t2i(0, 1)], extr_params[t2i(0, 2)], extr_params[t2i(0, 3)],
        extr_params[t2i(1, 0)], extr_params[t2i(1, 1)], extr_params[t2i(1, 2)], extr_params[t2i(1, 3)],
        extr_params[t2i(2, 0)], extr_params[t2i(2, 1)], extr_params[t2i(2, 2)], extr_params[t2i(2, 3)]
    };

    float cov3d_reg[6] = {
        cov3d_array[t2i(idx, 0, 6)], cov3d_array[t2i(idx, 1, 6)], cov3d_array[t2i(idx, 2, 6)],
        cov3d_array[t2i(idx, 3, 6)], cov3d_array[t2i(idx, 4, 6)], cov3d_array[t2i(idx, 5, 6)]
    };

    float3 dL_duvd_reg = float3(
        dL_duvd_array[3 * idx],
        dL_duvd_array[3 * idx + 1],
        dL_duvd_array[3 * idx + 2]
    );

    float3 dL_dcov2d_reg = float3(
        dL_dcov2d_array[3 * idx + 0],
        dL_dcov2d_array[3 * idx + 1],
        dL_dcov2d_array[3 * idx + 2]
    );

    float3 dL_dxyz_reg = { 0.0, 0.0, 0.0 };
    float dL_dintr_reg[4] = { 0.0 };
    float dL_dextr_reg[12] = { 0.0 };

    float3 xw = float3(
        xyz_array[3 * idx],
        xyz_array[3 * idx + 1],
        xyz_array[3 * idx + 2]
    );
    float3 xc = {
        extr_reg[0] * xw.x + extr_reg[1] * xw.y + extr_reg[2] * xw.z + extr_reg[3],
        extr_reg[4] * xw.x + extr_reg[5] * xw.y + extr_reg[6] * xw.z + extr_reg[7],
        extr_reg[8] * xw.x + extr_reg[9] * xw.y + extr_reg[10] * xw.z + extr_reg[11]
    };

    float norm = 1.f / xc.z;
    float norm2 = norm * norm;
    float norm3 = norm2 * norm;

    dL_dxyz_reg.x += (intr_reg[0] * (extr_reg[0] * xc.z - xc.x * extr_reg[8]) * norm2) * dL_duvd_reg.x;
    dL_dxyz_reg.x += (intr_reg[1] * (extr_reg[4] * xc.z - xc.y * extr_reg[8]) * norm2) * dL_duvd_reg.y;
    dL_dxyz_reg.x += extr_reg[8] * dL_duvd_reg.z;

    dL_dxyz_reg.y += (intr_reg[0] * (extr_reg[1] * xc.z - xc.x * extr_reg[9]) * norm2) * dL_duvd_reg.x;
    dL_dxyz_reg.y += (intr_reg[1] * (extr_reg[5] * xc.z - xc.y * extr_reg[9]) * norm2) * dL_duvd_reg.y;
    dL_dxyz_reg.y += extr_reg[9] * dL_duvd_reg.z;

    dL_dxyz_reg.z += (intr_reg[0] * (extr_reg[2] * xc.z - xc.x * extr_reg[10]) * norm2) * dL_duvd_reg.x;
    dL_dxyz_reg.z += (intr_reg[1] * (extr_reg[6] * xc.z - xc.y * extr_reg[10]) * norm2) * dL_duvd_reg.y;
    dL_dxyz_reg.z += extr_reg[10] * dL_duvd_reg.z;

    // if (dL_dintr_params.data_ptr() != nullptr) {
        dL_dintr_reg[0] += xc.x * norm * dL_duvd_reg.x;
        dL_dintr_reg[1] += xc.y * norm * dL_duvd_reg.y;
        dL_dintr_reg[2] += dL_duvd_reg.x;
        dL_dintr_reg[3] += dL_duvd_reg.y;
    // }

    // if (dL_dextr_params.data_ptr() != nullptr) {
        dL_dextr_reg[0] += intr_reg[0] * xw.x * norm * dL_duvd_reg.x;
        dL_dextr_reg[1] += intr_reg[0] * xw.y * norm * dL_duvd_reg.x;
        dL_dextr_reg[2] += intr_reg[0] * xw.z * norm * dL_duvd_reg.x;
        dL_dextr_reg[3] += intr_reg[0] * norm * dL_duvd_reg.x;

        dL_dextr_reg[4] += intr_reg[1] * xw.x * norm * dL_duvd_reg.y;
        dL_dextr_reg[5] += intr_reg[1] * xw.y * norm * dL_duvd_reg.y;
        dL_dextr_reg[6] += intr_reg[1] * xw.z * norm * dL_duvd_reg.y;
        dL_dextr_reg[7] += intr_reg[1] * norm * dL_duvd_reg.y;

        dL_dextr_reg[8] += -intr_reg[0] * xw.x * xc.x * norm2 * dL_duvd_reg.x;
        dL_dextr_reg[8] += -intr_reg[1] * xw.x * xc.y * norm2 * dL_duvd_reg.y;
        dL_dextr_reg[8] += xw.x * dL_duvd_reg.z;

        dL_dextr_reg[9] += -intr_reg[0] * xw.y * xc.x * norm2 * dL_duvd_reg.x;
        dL_dextr_reg[9] += -intr_reg[1] * xw.y * xc.y * norm2 * dL_duvd_reg.y;
        dL_dextr_reg[9] += xw.y * dL_duvd_reg.z;

        dL_dextr_reg[10] += -intr_reg[0] * xw.z * xc.x * norm2 * dL_duvd_reg.x;
        dL_dextr_reg[10] += -intr_reg[1] * xw.z * xc.y * norm2 * dL_duvd_reg.y;
        dL_dextr_reg[10] += xw.z * dL_duvd_reg.z;

        dL_dextr_reg[11] += -intr_reg[0] * xc.x * norm2 * dL_duvd_reg.x;
        dL_dextr_reg[11] += -intr_reg[1] * xc.y * norm2 * dL_duvd_reg.y;
        dL_dextr_reg[11] += dL_duvd_reg.z;
    // }

    float3x3 Rw2c = float3x3(
        extr_reg[0], extr_reg[1], extr_reg[2],
        extr_reg[4], extr_reg[5], extr_reg[6],
        extr_reg[8], extr_reg[9], extr_reg[10]
    );

    float3x3 Sigma = float3x3(
        cov3d_reg[0], cov3d_reg[1], cov3d_reg[2],
        cov3d_reg[1], cov3d_reg[3], cov3d_reg[4],
        cov3d_reg[2], cov3d_reg[4], cov3d_reg[5]
    );

    float2x3 J = float2x3(
        intr_reg[0] / xc.z, 0.0, -(intr_reg[0] * xc.x) / (xc.z * xc.z),
        0.0, intr_reg[1] / xc.z, -(intr_reg[1] * xc.y) / (xc.z * xc.z)
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

    // 1->1 dL_dT = dL_dcov * dcov_dT
    float dL_dT00, dL_dT10, dL_dT01, dL_dT11, dL_dT02, dL_dT12;
    dL_dT00 = dL_dT10 = dL_dT01 = dL_dT11 = dL_dT02 = dL_dT12 = 0;

    dL_dT00 += 2 * (cov3d_reg[0] * T[0][0] + cov3d_reg[1] * T[0][1] + cov3d_reg[2] * T[0][2]) * dL_dcov2d_reg.x;
    dL_dT00 += (cov3d_reg[0] * T[1][0] + cov3d_reg[1] * T[1][1] + cov3d_reg[2] * T[1][2]) * dL_dcov2d_reg.y;

    dL_dT01 += 2 * (cov3d_reg[1] * T[0][0] + cov3d_reg[3] * T[0][1] + cov3d_reg[4] * T[0][2]) * dL_dcov2d_reg.x;
    dL_dT01 += (cov3d_reg[1] * T[1][0] + cov3d_reg[3] * T[1][1] + cov3d_reg[4] * T[1][2]) * dL_dcov2d_reg.y;

    dL_dT02 += 2 * (cov3d_reg[2] * T[0][0] + cov3d_reg[4] * T[0][1] + cov3d_reg[5] * T[0][2]) * dL_dcov2d_reg.x;
    dL_dT02 += (cov3d_reg[2] * T[1][0] + cov3d_reg[4] * T[1][1] + cov3d_reg[5] * T[1][2]) * dL_dcov2d_reg.y;

    dL_dT10 += (cov3d_reg[0] * T[0][0] + cov3d_reg[1] * T[0][1] + cov3d_reg[2] * T[0][2]) * dL_dcov2d_reg.y;
    dL_dT10 += 2 * (cov3d_reg[0] * T[1][0] + cov3d_reg[1] * T[1][1] + cov3d_reg[2] * T[1][2]) * dL_dcov2d_reg.z;

    dL_dT11 += (cov3d_reg[1] * T[0][0] + cov3d_reg[3] * T[0][1] + cov3d_reg[4] * T[0][2]) * dL_dcov2d_reg.y;
    dL_dT11 += 2 * (cov3d_reg[1] * T[1][0] + cov3d_reg[3] * T[1][1] + cov3d_reg[4] * T[1][2]) * dL_dcov2d_reg.z;

    dL_dT12 += (cov3d_reg[2] * T[0][0] + cov3d_reg[4] * T[0][1] + cov3d_reg[5] * T[0][2]) * dL_dcov2d_reg.y;
    dL_dT12 += 2 * (cov3d_reg[2] * T[1][0] + cov3d_reg[4] * T[1][1] + cov3d_reg[5] * T[1][2]) * dL_dcov2d_reg.z;

    // 1->1->1 dL_dJ = dL_dT * dT_dJ
    float dL_dJ00, dL_dJ02, dL_dJ11, dL_dJ12;
    dL_dJ00 = dL_dT00 * extr_reg[0] + dL_dT01 * extr_reg[1] + dL_dT02 * extr_reg[2];
    dL_dJ02 = dL_dT00 * extr_reg[8] + dL_dT01 * extr_reg[9] + dL_dT02 * extr_reg[10];
    dL_dJ11 = dL_dT10 * extr_reg[4] + dL_dT11 * extr_reg[5] + dL_dT12 * extr_reg[6];
    dL_dJ12 = dL_dT10 * extr_reg[8] + dL_dT11 * extr_reg[9] + dL_dT12 * extr_reg[10];

    float dL_dxc, dL_dyc, dL_dzc;

    // 1->1->1->1 dL_dxc = dL_dJ * dJ_dxc
    dL_dxc = -dL_dJ02 * intr_reg[0] * norm2;
    dL_dyc = -dL_dJ12 * intr_reg[1] * norm2;
    dL_dzc = -dL_dJ00 * intr_reg[0] * norm2 + dL_dJ02 * 2 * intr_reg[0] * xc.x * norm3 -
              dL_dJ11 * intr_reg[1] * norm2 + dL_dJ12 * 2 * intr_reg[1] * xc.y * norm3;

    // if (dL_dintr_params.data_ptr() != nullptr) {
        dL_dintr_reg[0] += dL_dJ00 * norm - dL_dJ02 * xc.x * norm2;
        dL_dintr_reg[1] += dL_dJ11 * norm - dL_dJ12 * xc.y * norm2;

        float oldvalue = 0.0;
        dL_dintr_params.InterlockedAddF32(0, dL_dintr_reg[0], oldvalue);
        dL_dintr_params.InterlockedAddF32(1, dL_dintr_reg[1], oldvalue);
        dL_dintr_params.InterlockedAddF32(2, dL_dintr_reg[2], oldvalue);
        dL_dintr_params.InterlockedAddF32(3, dL_dintr_reg[3], oldvalue);
    // }

    // if (dL_dextr_params.data_ptr() != nullptr) {
        // 1->1->2 dL_dW = dL_dT * dT_dW
        dL_dextr_reg[0] += dL_dT00 * J[0][0];
        dL_dextr_reg[1] += dL_dT01 * J[0][0];
        dL_dextr_reg[2] += dL_dT02 * J[0][0];

        dL_dextr_reg[4] += dL_dT10 * J[1][1];
        dL_dextr_reg[5] += dL_dT11 * J[1][1];
        dL_dextr_reg[6] += dL_dT12 * J[1][1];

        dL_dextr_reg[8] += dL_dT00 * J[0][2] + dL_dT10 * J[1][2];
        dL_dextr_reg[9] += dL_dT01 * J[0][2] + dL_dT11 * J[1][2];
        dL_dextr_reg[10] += dL_dT02 * J[0][2] + dL_dT12 * J[1][2];

        // 1->1->1->1->1 dL_dextr = dL_dxc * dxc_dextr
        dL_dextr_reg[0] += dL_dxc * xw.x;
        dL_dextr_reg[1] += dL_dxc * xw.y;
        dL_dextr_reg[2] += dL_dxc * xw.z;
        dL_dextr_reg[3] += dL_dxc;

        dL_dextr_reg[4] += dL_dyc * xw.x;
        dL_dextr_reg[5] += dL_dyc * xw.y;
        dL_dextr_reg[6] += dL_dyc * xw.z;
        dL_dextr_reg[7] += dL_dyc;

        dL_dextr_reg[8] += dL_dzc * xw.x;
        dL_dextr_reg[9] += dL_dzc * xw.y;
        dL_dextr_reg[10] += dL_dzc * xw.z;
        dL_dextr_reg[11] += dL_dzc;

        // float oldvalue = 0.0;
        dL_dextr_params.InterlockedAddF32(t2i(0, 0), dL_dextr_reg[0], oldvalue);
        dL_dextr_params.InterlockedAddF32(t2i(0, 1), dL_dextr_reg[1], oldvalue);
        dL_dextr_params.InterlockedAddF32(t2i(0, 2), dL_dextr_reg[2], oldvalue);
        dL_dextr_params.InterlockedAddF32(t2i(0, 3), dL_dextr_reg[3], oldvalue);

        dL_dextr_params.InterlockedAddF32(t2i(1, 0), dL_dextr_reg[4], oldvalue);
        dL_dextr_params.InterlockedAddF32(t2i(1, 1), dL_dextr_reg[5], oldvalue);
        dL_dextr_params.InterlockedAddF32(t2i(1, 2), dL_dextr_reg[6], oldvalue);
        dL_dextr_params.InterlockedAddF32(t2i(1, 3), dL_dextr_reg[7], oldvalue);

        dL_dextr_params.InterlockedAddF32(t2i(2, 0), dL_dextr_reg[8], oldvalue);
        dL_dextr_params.InterlockedAddF32(t2i(2, 1), dL_dextr_reg[9], oldvalue);
        dL_dextr_params.InterlockedAddF32(t2i(2, 2), dL_dextr_reg[10], oldvalue);
        dL_dextr_params.InterlockedAddF32(t2i(2, 3), dL_dextr_reg[11], oldvalue);
    // }

    // 1->1->1->1->2 dL_dxyz = dL_dxc * dxc_dxw
    dL_dxyz_reg.x += dL_dxc * extr_reg[0] + dL_dyc * extr_reg[4] + dL_dzc * extr_reg[8];
    dL_dxyz_reg.y += dL_dxc * extr_reg[1] + dL_dyc * extr_reg[5] + dL_dzc * extr_reg[9];
    dL_dxyz_reg.z += dL_dxc * extr_reg[2] + dL_dyc * extr_reg[6] + dL_dzc * extr_reg[10];

    dL_dxyz_array[3 * idx] = dL_dxyz_reg.x;
    dL_dxyz_array[3 * idx + 1] = dL_dxyz_reg.y;
    dL_dxyz_array[3 * idx + 2] = dL_dxyz_reg.z;

    // 1->2 dL_dcov3d = dL_dcov * dcov_dcov3d
    float dL_dcov3d[6] = { 0 };
    dL_dcov3d[0] = T[0][0] * T[0][0] * dL_dcov2d_reg.x +
                   T[0][0] * T[1][0] * dL_dcov2d_reg.y +
                   T[1][0] * T[1][0] * dL_dcov2d_reg.z;

    dL_dcov3d[1] = 2 * T[0][0] * T[0][1] * dL_dcov2d_reg.x +
                    (T[1][0] * T[0][1] + T[0][0] * T[1][1]) * dL_dcov2d_reg.y +
                    2 * T[1][0] * T[1][1] * dL_dcov2d_reg.z;

    dL_dcov3d[2] = 2 * T[0][0] * T[0][2] * dL_dcov2d_reg.x +
                    (T[0][0] * T[1][2] + T[1][0] * T[0][2]) * dL_dcov2d_reg.y +
                    2 * T[1][0] * T[1][2] * dL_dcov2d_reg.z;

    dL_dcov3d[3] = T[0][1] * T[0][1] * dL_dcov2d_reg.x +
                    T[0][1] * T[1][1] * dL_dcov2d_reg.y +
                    T[1][1] * T[1][1] * dL_dcov2d_reg.z;

    dL_dcov3d[4] = 2 * T[0][1] * T[0][2] * dL_dcov2d_reg.x +
                    (T[0][1] * T[1][2] + T[1][1] * T[0][2]) * dL_dcov2d_reg.y +
                    2 * T[1][1] * T[1][2] * dL_dcov2d_reg.z;

    dL_dcov3d[5] = T[0][2] * T[0][2] * dL_dcov2d_reg.x +
                    T[0][2] * T[1][2] * dL_dcov2d_reg.y +
                    T[1][2] * T[1][2] * dL_dcov2d_reg.z;

    float3x3 S = scale_vector_to_matrix(scale_act_reg);
    float3x3 R = unit_quaternion_to_rotmatrix(rotate_act_reg);
    float3x3 RS = mul(R, S);

    float3x3 dL_dSigma = float3x3(
        dL_dcov3d[0], 0.5f * dL_dcov3d[1], 0.5f * dL_dcov3d[2],
        0.5f * dL_dcov3d[1], dL_dcov3d[3], 0.5f * dL_dcov3d[4],
        0.5f * dL_dcov3d[2], 0.5f * dL_dcov3d[4], dL_dcov3d[5]
    );
    float3x3 dL_dRS = 2.0f * mul(dL_dSigma, RS);

    // 1->2->1->1. dL_dscale_array = dL_dcov3d * dcov3d_dscale
    float3 dL_dscale_act_reg = float3(
        dL_dRS[0][0] * R[0][0] + dL_dRS[1][0] * R[1][0] + dL_dRS[2][0] * R[2][0],
        dL_dRS[0][1] * R[0][1] + dL_dRS[1][1] * R[1][1] + dL_dRS[2][1] * R[2][1],
        dL_dRS[0][2] * R[0][2] + dL_dRS[1][2] * R[1][2] + dL_dRS[2][2] * R[2][2]
    );
    dL_dscale_array[3 * idx] = dL_dscale_act_reg.x * scale_act_reg.x;
    dL_dscale_array[3 * idx + 1] = dL_dscale_act_reg.y * scale_act_reg.y;
    dL_dscale_array[3 * idx + 2] = dL_dscale_act_reg.z * scale_act_reg.z;

    // 1->2->1->2. dL_drotate = dL_dRS * dRS_drotate (rotate matrix)
    dL_dRS[0][0] *= scale_act_reg.x; dL_dRS[1][0] *= scale_act_reg.x; dL_dRS[2][0] *= scale_act_reg.x;
    dL_dRS[0][1] *= scale_act_reg.y; dL_dRS[1][1] *= scale_act_reg.y; dL_dRS[2][1] *= scale_act_reg.y;
    dL_dRS[0][2] *= scale_act_reg.z; dL_dRS[1][2] *= scale_act_reg.z; dL_dRS[2][2] *= scale_act_reg.z;

    // 1->2->1->2->1 dL_dquat = dL_drotate * drotate_dquat
    // keep in mid that we use WXYZ! But it's stored in xyzw (float4).
    float w = rotate_act_reg[0];
    float x = rotate_act_reg[1];
    float y = rotate_act_reg[2];
    float z = rotate_act_reg[3];

    float dL_dqw_act = 2 * x * (dL_dRS[2][1] - dL_dRS[1][2]) +
                    2 * y * (dL_dRS[0][2] - dL_dRS[2][0]) +
                    2 * z * (dL_dRS[1][0] - dL_dRS[0][1]);

    float dL_dqx_act = 2 * w * (dL_dRS[2][1] - dL_dRS[1][2]) -
                    4 * x * (dL_dRS[1][1] + dL_dRS[2][2]) +
                    2 * y * (dL_dRS[1][0] + dL_dRS[0][1]) +
                    2 * z * (dL_dRS[2][0] + dL_dRS[0][2]);

    float dL_dqy_act = 2 * w * (dL_dRS[0][2] - dL_dRS[2][0]) +
                    2 * x * (dL_dRS[1][0] + dL_dRS[0][1]) -
                    4 * y * (dL_dRS[0][0] + dL_dRS[2][2]) +
                    2 * z * (dL_dRS[2][1] + dL_dRS[1][2]);

    float dL_dqz_act = 2 * w * (dL_dRS[1][0] - dL_dRS[0][1]) +
                    2 * x * (dL_dRS[2][0] + dL_dRS[0][2]) +
                    2 * y * (dL_dRS[2][1] + dL_dRS[1][2]) -
                    4 * z * (dL_dRS[0][0] + dL_dRS[1][1]);

    // keep in mid that we use WXYZ! But it's stored in xyzw (float4).
    float norm_rotate3 = pow(norm_rotate, 3);
    float dL_dqw = (1 / norm_rotate - rotate_reg[0] * rotate_reg[0] / norm_rotate3) * dL_dqw_act +
                   (-rotate_reg[0] * rotate_reg[1] / norm_rotate3) * dL_dqx_act +
                   (-rotate_reg[0] * rotate_reg[2] / norm_rotate3) * dL_dqy_act +
                   (-rotate_reg[0] * rotate_reg[3] / norm_rotate3) * dL_dqz_act;
    float dL_dqx = (-rotate_reg[1] * rotate_reg[0] / norm_rotate3) * dL_dqw_act +
                   (1 / norm_rotate - rotate_reg[1] * rotate_reg[1] / norm_rotate3) * dL_dqx_act +
                   (-rotate_reg[1] * rotate_reg[2] / norm_rotate3) * dL_dqy_act +
                   (-rotate_reg[1] * rotate_reg[3] / norm_rotate3) * dL_dqz_act;
    float dL_dqy = (-rotate_reg[2] * rotate_reg[0] / norm_rotate3) * dL_dqw_act +
                   (-rotate_reg[2] * rotate_reg[1] / norm_rotate3) * dL_dqx_act +
                   (1 / norm_rotate - rotate_reg[2] * rotate_reg[2] / norm_rotate3) * dL_dqy_act +
                   (-rotate_reg[2] * rotate_reg[3] / norm_rotate3) * dL_dqz_act;
    float dL_dqz = (-rotate_reg[3] * rotate_reg[0] / norm_rotate3) * dL_dqw_act +
                   (-rotate_reg[3] * rotate_reg[1] / norm_rotate3) * dL_dqx_act +
                   (-rotate_reg[3] * rotate_reg[2] / norm_rotate3) * dL_dqy_act +
                   (1 / norm_rotate - rotate_reg[3] * rotate_reg[3] / norm_rotate3) * dL_dqz_act;
    float4 dL_drotate_reg = {
        dL_dqw, dL_dqx, dL_dqy, dL_dqz
    };
    dL_drotate_array[idx] = dL_drotate_reg;
}