# include "utils.slang"
[CudaKernel]
void EvalSHBackwardKernel(
    // inputs-scalars
    int P,
    int D,
    // inputs-buffers
    TensorView<float> sh_array,                   // [P, 3, D]
    TensorView<float3> xyz_array,           // [P, 3]
    TensorView<float> extr_params,
    TensorView<bool> visible_array,               // [P, 1]
    TensorView<float3> dL_dval_array,             // [P, 3]
    // outputs
    TensorView<float> dL_dshs_array,              // [P, 3, D]
    TensorView<float3> dL_dxyz_array,           // [P, 3]
    TensorView<float> dL_dextr_params
)
{
}

static const float SH_C0 = 0.28209479177387814f;

static const float SH_C1[] = {
    -0.4886025119029199f,
    0.4886025119029199f
};

static const float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f
};

static const float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f
};

void set_sh_grad_value(
    RWStructuredBuffer<float> dL_dshs_array,
    uint32_t idx,
    uint32_t coeff_id,
    float3 dL_dshs_tmp
)
{
    dL_dshs_array[t2i(idx, 0, coeff_id, 3, DD)] = dL_dshs_tmp.x;
    dL_dshs_array[t2i(idx, 1, coeff_id, 3, DD)] = dL_dshs_tmp.y;
    dL_dshs_array[t2i(idx, 2, coeff_id, 3, DD)] = dL_dshs_tmp.z;
}

// Buffer bindings
[[vk::binding(0)]] StructuredBuffer<float> sh_coeff_array;
[[vk::binding(1)]] StructuredBuffer<float> xyz_array;
[[vk::binding(2)]] StructuredBuffer<float> extr_params;

[[vk::binding(3)]] StructuredBuffer<uint32_t> visible_array;
[[vk::binding(4)]] RWStructuredBuffer<float> value_array;

[[vk::binding(5)]] StructuredBuffer<float> dL_dval_array;
[[vk::binding(6)]] RWStructuredBuffer<float> dL_dshs_array;
[[vk::binding(7)]] RWStructuredBuffer<float> dL_dxyz_array;
[[vk::binding(8)]] RWByteAddressBuffer dL_dextr_params;

[[vk::binding(9)]] StructuredBuffer<int32_t> curPointsNum;

[[vk::binding(10)]] RWStructuredBuffer<float> extr;
[[vk::binding(11)]] RWStructuredBuffer<float> camera_center;
[[vk::binding(12)]] RWStructuredBuffer<float> extr_act;
[[vk::binding(13)]] StructuredBuffer<uint32_t> frozen;

[[vk::push_constant]]
cbuffer PushConstants
{
    float xyz_size;
    float width;
    float height;
    float position_lr;
    float D;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint32_t idx = dispatchThreadID.x;
    if (idx >= curPointsNum[0] || visible_array[idx] == 0 || frozen[idx] == 1)
        return;

    float3 dir_un = {
        xyz_array[3 * idx + 0] - camera_center[0],
        xyz_array[3 * idx + 1] - camera_center[1],
        xyz_array[3 * idx + 2] - camera_center[2],
    };

    float norm = sqrt(dir_un.x * dir_un.x + dir_un.y * dir_un.y + dir_un.z * dir_un.z);
    float3 dir = dir_un / norm;

    float3 dL_dval_tmp = float3(
        dL_dval_array[idx * 3 + 0],
        dL_dval_array[idx * 3 + 1],
        dL_dval_array[idx * 3 + 2]
    );

    if (value_array[idx * 3 + 0] <= 0)
        dL_dval_tmp.x = 0;
    if (value_array[idx * 3 + 1] <= 0)
        dL_dval_tmp.y = 0;
    if (value_array[idx * 3 + 2] <= 0)
        dL_dval_tmp.z = 0;

    float3 dL_dvdirs = float3(0, 0, 0);

    // SH0
    float3 coeff0 = float3(sh_coeff_array[t2i(idx, 0, 0, 3, DD)], sh_coeff_array[t2i(idx, 1, 0, 3, DD)], sh_coeff_array[t2i(idx, 2, 0, 3, DD)]);
    float3 dL_dshs_tmp = dL_dval_tmp * SH_C0;
    set_sh_grad_value(dL_dshs_array, idx, 0, dL_dshs_tmp);

    if (D > 1) {
        // SH1
        float3 coeff1 = float3(sh_coeff_array[t2i(idx, 0, 1, 3, DD)], sh_coeff_array[t2i(idx, 1, 1, 3, DD)], sh_coeff_array[t2i(idx, 2, 1, 3, DD)]);
        float3 coeff2 = float3(sh_coeff_array[t2i(idx, 0, 2, 3, DD)], sh_coeff_array[t2i(idx, 1, 2, 3, DD)], sh_coeff_array[t2i(idx, 2, 2, 3, DD)]);
        float3 coeff3 = float3(sh_coeff_array[t2i(idx, 0, 3, 3, DD)], sh_coeff_array[t2i(idx, 1, 3, 3, DD)], sh_coeff_array[t2i(idx, 2, 3, 3, DD)]);
        float x = dir.x, y = dir.y, z = dir.z;

        dL_dshs_tmp = dL_dval_tmp * SH_C1[0] * y;
        set_sh_grad_value(dL_dshs_array, idx, 1, dL_dshs_tmp);

        dL_dshs_tmp = dL_dval_tmp * SH_C1[1] * z;
        set_sh_grad_value(dL_dshs_array, idx, 2, dL_dshs_tmp);

        dL_dshs_tmp = dL_dval_tmp * SH_C1[0] * x;
        set_sh_grad_value(dL_dshs_array, idx, 3, dL_dshs_tmp);

        float3 dL_ddirx_tmp = SH_C1[0] * coeff3 * dL_dval_tmp;
        dL_dvdirs.x += dL_ddirx_tmp.x + dL_ddirx_tmp.y + dL_ddirx_tmp.z;

        float3 dL_ddiry_tmp = SH_C1[0] * coeff1 * dL_dval_tmp;
        dL_dvdirs.y += dL_ddiry_tmp.x + dL_ddiry_tmp.y + dL_ddiry_tmp.z;

        float3 dL_ddirz_tmp = SH_C1[1] * coeff2 * dL_dval_tmp;
        dL_dvdirs.z += dL_ddirz_tmp.x + dL_ddirz_tmp.y + dL_ddirz_tmp.z;

        if (D > 4) {
            // SH2
            float x2 = x * x, y2 = y * y, z2 = z * z;
            float xy = x * y, yz = y * z, xz = x * z;

            float3 coeff4 = float3(sh_coeff_array[t2i(idx, 0, 4, 3, DD)], sh_coeff_array[t2i(idx, 1, 4, 3, DD)], sh_coeff_array[t2i(idx, 2, 4, 3, DD)]);
            float3 coeff5 = float3(sh_coeff_array[t2i(idx, 0, 5, 3, DD)], sh_coeff_array[t2i(idx, 1, 5, 3, DD)], sh_coeff_array[t2i(idx, 2, 5, 3, DD)]);
            float3 coeff6 = float3(sh_coeff_array[t2i(idx, 0, 6, 3, DD)], sh_coeff_array[t2i(idx, 1, 6, 3, DD)], sh_coeff_array[t2i(idx, 2, 6, 3, DD)]);
            float3 coeff7 = float3(sh_coeff_array[t2i(idx, 0, 7, 3, DD)], sh_coeff_array[t2i(idx, 1, 7, 3, DD)], sh_coeff_array[t2i(idx, 2, 7, 3, DD)]);
            float3 coeff8 = float3(sh_coeff_array[t2i(idx, 0, 8, 3, DD)], sh_coeff_array[t2i(idx, 1, 8, 3, DD)], sh_coeff_array[t2i(idx, 2, 8, 3, DD)]);

            dL_dshs_tmp = dL_dval_tmp * SH_C2[0] * xy;
            set_sh_grad_value(dL_dshs_array, idx, 4, dL_dshs_tmp);
            dL_dshs_tmp = dL_dval_tmp * SH_C2[1] * yz;
            set_sh_grad_value(dL_dshs_array, idx, 5, dL_dshs_tmp);
            dL_dshs_tmp = dL_dval_tmp * SH_C2[2] * (2.0f * z2 - x2 - y2);
            set_sh_grad_value(dL_dshs_array, idx, 6, dL_dshs_tmp);
            dL_dshs_tmp = dL_dval_tmp * SH_C2[1] * xz;
            set_sh_grad_value(dL_dshs_array, idx, 7, dL_dshs_tmp);
            dL_dshs_tmp = dL_dval_tmp * SH_C2[0] * 0.5 * (x2 - y2);
            set_sh_grad_value(dL_dshs_array, idx, 8, dL_dshs_tmp);

            dL_ddirx_tmp = (SH_C2[0] * coeff4 * y +
                            SH_C2[2] * coeff6 * (-2 * x) +
                            SH_C2[1] * coeff7 * z +
                            SH_C2[0] * coeff8 * x) * dL_dval_tmp;

            dL_dvdirs.x += dL_ddirx_tmp.x + dL_ddirx_tmp.y + dL_ddirx_tmp.z;

            dL_ddiry_tmp = (SH_C2[0] * coeff4 * x +
                            SH_C2[1] * coeff5 * z +
                            SH_C2[2] * coeff6 * (-2 * y) +
                            SH_C2[0] * coeff8 * (-y)
                            ) * dL_dval_tmp;
            dL_dvdirs.y += dL_ddiry_tmp.x + dL_ddiry_tmp.y + dL_ddiry_tmp.z;

            dL_ddirz_tmp = (SH_C2[1] * coeff5 * y +
                            SH_C2[2] * coeff6 * 4 * z +
                            SH_C2[1] * coeff7 * x) * dL_dval_tmp;
            dL_dvdirs.z += dL_ddirz_tmp.x + dL_ddirz_tmp.y + dL_ddirz_tmp.z;

            if (D > 9) {
                // SH3
                float3 coeff9 = float3(sh_coeff_array[t2i(idx, 0, 9, 3, DD)], sh_coeff_array[t2i(idx, 1, 9, 3, DD)], sh_coeff_array[t2i(idx, 2, 9, 3, DD)]);
                float3 coeff10 = float3(sh_coeff_array[t2i(idx, 0, 10, 3, DD)], sh_coeff_array[t2i(idx, 1, 10, 3, DD)], sh_coeff_array[t2i(idx, 2, 10, 3, DD)]);
                float3 coeff11 = float3(sh_coeff_array[t2i(idx, 0, 11, 3, DD)], sh_coeff_array[t2i(idx, 1, 11, 3, DD)], sh_coeff_array[t2i(idx, 2, 11, 3, DD)]);
                float3 coeff12 = float3(sh_coeff_array[t2i(idx, 0, 12, 3, DD)], sh_coeff_array[t2i(idx, 1, 12, 3, DD)], sh_coeff_array[t2i(idx, 2, 12, 3, DD)]);
                float3 coeff13 = float3(sh_coeff_array[t2i(idx, 0, 13, 3, DD)], sh_coeff_array[t2i(idx, 1, 13, 3, DD)], sh_coeff_array[t2i(idx, 2, 13, 3, DD)]);
                float3 coeff14 = float3(sh_coeff_array[t2i(idx, 0, 14, 3, DD)], sh_coeff_array[t2i(idx, 1, 14, 3, DD)], sh_coeff_array[t2i(idx, 2, 14, 3, DD)]);
                float3 coeff15 = float3(sh_coeff_array[t2i(idx, 0, 15, 3, DD)], sh_coeff_array[t2i(idx, 1, 15, 3, DD)], sh_coeff_array[t2i(idx, 2, 15, 3, DD)]);

                dL_dshs_tmp = dL_dval_tmp * SH_C3[0] * y * (3.0f * x2 - y2);
                set_sh_grad_value(dL_dshs_array, idx, 9, dL_dshs_tmp);
                dL_dshs_tmp = dL_dval_tmp * SH_C3[1] * xy * z;
                set_sh_grad_value(dL_dshs_array, idx, 10, dL_dshs_tmp);
                dL_dshs_tmp = dL_dval_tmp * SH_C3[2] * y * (4.0f * z2 - x2 - y2);
                set_sh_grad_value(dL_dshs_array, idx, 11, dL_dshs_tmp);
                dL_dshs_tmp = dL_dval_tmp * SH_C3[3] * z * (2.0f * z2 - 3.0f * x2 - 3.0f * y2);
                set_sh_grad_value(dL_dshs_array, idx, 12, dL_dshs_tmp);
                dL_dshs_tmp = dL_dval_tmp * SH_C3[2] * x * (4.0f * z2 - x2 - y2);
                set_sh_grad_value(dL_dshs_array, idx, 13, dL_dshs_tmp);
                dL_dshs_tmp = dL_dval_tmp * SH_C3[1] * z * 0.5 * (x2 - y2);
                set_sh_grad_value(dL_dshs_array, idx, 14, dL_dshs_tmp);
                dL_dshs_tmp = dL_dval_tmp * SH_C3[0] * x * (x2 - 3.0f * y2);
                set_sh_grad_value(dL_dshs_array, idx, 15, dL_dshs_tmp);

                dL_ddirx_tmp = (
                    SH_C3[0] * coeff9 * 6 * xy +
                                SH_C3[1] * coeff10 * yz +
                                SH_C3[2] * coeff11 * (-2 * xy) +
                                SH_C3[3] * coeff12 * (-6 * xz) +
                                SH_C3[2] * coeff13 * (4 * z2 - 3 * x2 - y2) +
                                SH_C3[1] * coeff14 * xz +
                                SH_C3[0] * coeff15 * (3 * x2 - 3 * y2)
                ) * dL_dval_tmp;
                dL_dvdirs.x += dL_ddirx_tmp.x + dL_ddirx_tmp.y + dL_ddirx_tmp.z;

                dL_ddiry_tmp = (
                    SH_C3[0] * coeff9 * (3 * x2 - 3 * y2) +
                                SH_C3[1] * coeff10 * xz +
                                SH_C3[2] * coeff11 * (4 * z2 - x2 - 3 * y2) +
                                SH_C3[3] * coeff12 * (-6.0 * yz) +
                                SH_C3[2] * coeff13 * (-2.0 * xy) +
                                SH_C3[1] * coeff14 * (-yz) +
                                SH_C3[0] * coeff15 * (-6.0 * xy)
                ) * dL_dval_tmp;
                dL_dvdirs.y += dL_ddiry_tmp.x + dL_ddiry_tmp.y + dL_ddiry_tmp.z;

                dL_ddirz_tmp = (
                    SH_C3[1] * coeff10 * xy +
                                SH_C3[2] * coeff11 * 8 * yz +
                                SH_C3[3] * coeff12 * (6 * z2 - 3 * x2 - 3 * y2) +
                                SH_C3[2] * coeff13 * 8 * xz +
                                SH_C3[1] * coeff14 * 0.5 * (x2 - y2) 
                ) * dL_dval_tmp;
                dL_dvdirs.z += dL_ddirz_tmp.x + dL_ddirz_tmp.y + dL_ddirz_tmp.z;
            }
        }
    }

    // if D <= 1, the final feature is not related to the direction.
    if (D <= 1) return;
    // dL_dvdirs_un = dL_dvdir * dvdirs_dvdirs_un

    float norm2 = norm * norm;
    float norm3 = norm2 * norm;

    float un_x = dir_un.x;
    float un_y = dir_un.y;
    float un_z = dir_un.z;
    float un_xx = un_x * un_x;
    float un_yy = un_y * un_y;
    float un_zz = un_z * un_z;
    float un_xy = un_x * un_y;
    float un_yz = un_y * un_z;
    float un_xz = un_x * un_z;

    float3 dL_dvdirs_un = {
        (norm2 - un_xx) / norm3 * dL_dvdirs.x + (-un_xy) / norm3 * dL_dvdirs.y + (-un_xz) / norm3 * dL_dvdirs.z,
        (-un_xy) / norm3 * dL_dvdirs.x + (norm2 - un_yy) / norm3 * dL_dvdirs.y + (-un_yz) / norm3 * dL_dvdirs.z,
        (-un_xz) / norm3 * dL_dvdirs.x + (-un_yz) / norm3 * dL_dvdirs.y + (norm2 - un_zz) / norm3 * dL_dvdirs.z,
    };

    // 1. dL_dxyz = dL_dvdirs_un * dvdirs_un_dxyz
    dL_dxyz_array[3 * idx + 0] += dL_dvdirs_un.x;
    dL_dxyz_array[3 * idx + 1] += dL_dvdirs_un.y;
    dL_dxyz_array[3 * idx + 2] += dL_dvdirs_un.z;

    // 2. dL_dextr = dL_dvdirs_un * dvdirs_un_dextr
    // 两个减法抵消
    // float dL_dextr0 = extr[3] * dL_dvdirs_un.x;
    // float dL_dextr1 = extr[3] * dL_dvdirs_un.y;
    // float dL_dextr2 = extr[3] * dL_dvdirs_un.z;
    // float dL_dextr3 = extr[0] * dL_dvdirs_un.x + extr[1] * dL_dvdirs_un.y + extr[2] * dL_dvdirs_un.z;

    // float dL_dextr4 = extr[7] * dL_dvdirs_un.x;
    // float dL_dextr5 = extr[7] * dL_dvdirs_un.y;
    // float dL_dextr6 = extr[7] * dL_dvdirs_un.z;
    // float dL_dextr7 = extr[4] * dL_dvdirs_un.x + extr[5] * dL_dvdirs_un.y + extr[6] * dL_dvdirs_un.z;

    // float dL_dextr8 = extr[11] * dL_dvdirs_un.x;
    // float dL_dextr9 = extr[11] * dL_dvdirs_un.y;
    // float dL_dextr10 = extr[11] * dL_dvdirs_un.z;
    // float dL_dextr11 = extr[8] * dL_dvdirs_un.x + extr[9] * dL_dvdirs_un.y + extr[10] * dL_dvdirs_un.z;

    float dL_dextr_reg[12] = {
        extr[3] * dL_dvdirs_un.x,
        extr[3] * dL_dvdirs_un.y,
        extr[3] * dL_dvdirs_un.z,
        extr[0] * dL_dvdirs_un.x + extr[1] * dL_dvdirs_un.y + extr[2] * dL_dvdirs_un.z,
        extr[7] * dL_dvdirs_un.x,
        extr[7] * dL_dvdirs_un.y,
        extr[7] * dL_dvdirs_un.z,
        extr[4] * dL_dvdirs_un.x + extr[5] * dL_dvdirs_un.y + extr[6] * dL_dvdirs_un.z,
        extr[11] * dL_dvdirs_un.x,
        extr[11] * dL_dvdirs_un.y,
        extr[11] * dL_dvdirs_un.z,
        extr[8] * dL_dvdirs_un.x + extr[9] * dL_dvdirs_un.y + extr[10] * dL_dvdirs_un.z
    };

    float oldvalue = 0;
    {
        float we = extr_act[0];
        float xe = extr_act[1];
        float ye = extr_act[2];
        float ze = extr_act[3];
        float norm_e = extr_act[4];
        float norm_e3 = extr_act[5];

        float dL_dqwe_act = 2 * xe * (dL_dextr_reg[9] - dL_dextr_reg[6]) +
                            2 * ye * (dL_dextr_reg[2] - dL_dextr_reg[8]) +
                            2 * ze * (dL_dextr_reg[4] - dL_dextr_reg[1]);

        float dL_dqxe_act = 2 * we * (dL_dextr_reg[9] - dL_dextr_reg[6]) -
                            4 * xe * (dL_dextr_reg[5] + dL_dextr_reg[10]) +
                            2 * ye * (dL_dextr_reg[4] + dL_dextr_reg[1]) +
                            2 * ze * (dL_dextr_reg[8] + dL_dextr_reg[2]);

        float dL_dqye_act = 2 * we * (dL_dextr_reg[2] - dL_dextr_reg[8]) +
                            2 * xe * (dL_dextr_reg[4] + dL_dextr_reg[1]) -
                            4 * ye * (dL_dextr_reg[0] + dL_dextr_reg[10]) +
                            2 * ze * (dL_dextr_reg[9] + dL_dextr_reg[6]);

        float dL_dqze_act = 2 * we * (dL_dextr_reg[4] - dL_dextr_reg[1]) +
                            2 * xe * (dL_dextr_reg[8] + dL_dextr_reg[2]) +
                            2 * ye * (dL_dextr_reg[9] + dL_dextr_reg[6]) -
                            4 * ze * (dL_dextr_reg[0] + dL_dextr_reg[5]);

        float dL_dqwe = (1 / norm_e - extr_params[0] * extr_params[0] / norm_e3) * dL_dqwe_act +
                        (-extr_params[0] * extr_params[1] / norm_e3) * dL_dqxe_act +
                        (-extr_params[0] * extr_params[2] / norm_e3) * dL_dqye_act +
                        (-extr_params[0] * extr_params[3] / norm_e3) * dL_dqze_act;
        float dL_dqxe = (-extr_params[1] * extr_params[0] / norm_e3) * dL_dqwe_act +
                        (1 / norm_e - extr_params[1] * extr_params[1] / norm_e3) * dL_dqxe_act +
                        (-extr_params[1] * extr_params[2] / norm_e3) * dL_dqye_act +
                        (-extr_params[1] * extr_params[3] / norm_e3) * dL_dqze_act;
        float dL_dqye = (-extr_params[2] * extr_params[0] / norm_e3) * dL_dqwe_act +
                        (-extr_params[2] * extr_params[1] / norm_e3) * dL_dqxe_act +
                        (1 / norm_e - extr_params[2] * extr_params[2] / norm_e3) * dL_dqye_act +
                        (-extr_params[2] * extr_params[3] / norm_e3) * dL_dqze_act;
        float dL_dqze = (-extr_params[3] * extr_params[0] / norm_e3) * dL_dqwe_act +
                        (-extr_params[3] * extr_params[1] / norm_e3) * dL_dqxe_act +
                        (-extr_params[3] * extr_params[2] / norm_e3) * dL_dqye_act +
                        (1 / norm_e - extr_params[3] * extr_params[3] / norm_e3) * dL_dqze_act;

        dL_dextr_params.InterlockedAddF32(0 * sizeof(float), dL_dqwe, oldvalue);
        dL_dextr_params.InterlockedAddF32(1 * sizeof(float), dL_dqxe, oldvalue);
        dL_dextr_params.InterlockedAddF32(2 * sizeof(float), dL_dqye, oldvalue);
        dL_dextr_params.InterlockedAddF32(3 * sizeof(float), dL_dqze, oldvalue);
        dL_dextr_params.InterlockedAddF32(4 * sizeof(float), dL_dextr_reg[3], oldvalue);
        dL_dextr_params.InterlockedAddF32(5 * sizeof(float), dL_dextr_reg[7], oldvalue);
        dL_dextr_params.InterlockedAddF32(6 * sizeof(float), dL_dextr_reg[11], oldvalue);
    }
}