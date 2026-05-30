# include "utils.slang"

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

// Buffer bindings
[[vk::binding(0)]] StructuredBuffer<float> sh_coeff_array;
[[vk::binding(1)]] StructuredBuffer<float> xyz_array;
[[vk::binding(2)]] StructuredBuffer<float> extr_params;

[[vk::binding(3)]] StructuredBuffer<uint32_t> visible_array;
[[vk::binding(4)]] RWStructuredBuffer<float> value_array;

[[vk::binding(9)]] StructuredBuffer<int32_t> curPointsNum;

[[vk::binding(10)]] RWStructuredBuffer<float> extr;
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

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint32_t idx = dispatchThreadID.x;
    if (idx >= curPointsNum[0])
        return;
    
    // AllMemoryBarrierWithGroupSync();
    // if (idx == 0) {
    //     camera_center[0] = -(extr[0] * extr[3] + extr[4] * extr[7] + extr[8] * extr[11]);
    //     camera_center[1] = -(extr[1] * extr[3] + extr[5] * extr[7] + extr[9] * extr[11]);
    //     camera_center[2] = -(extr[2] * extr[3] + extr[6] * extr[7] + extr[10] * extr[11]);
    // }
    // AllMemoryBarrierWithGroupSync();

    if (visible_array[idx] == 0)
        return;
    

    float3 dir = {
        xyz_array[3 * idx + 0] - camera_center[0],
        xyz_array[3 * idx + 1] - camera_center[1],
        xyz_array[3 * idx + 2] - camera_center[2],
    };
    dir = dir / sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);

    // SH0
    float3 coeff0 = float3(sh_coeff_array[idx * 3 + 0], sh_coeff_array[idx * 3 + 1], sh_coeff_array[idx * 3 + 2]);
    float3 rgb = SH_C0 * coeff0;

    /*
    if (D > 1) {
        // SH1
        float3 coeff1 = float3(sh_coeff_array[t2i(idx, 0, 1, 3, DD)], sh_coeff_array[t2i(idx, 1, 1, 3, DD)], sh_coeff_array[t2i(idx, 2, 1, 3, DD)]);
        float3 coeff2 = float3(sh_coeff_array[t2i(idx, 0, 2, 3, DD)], sh_coeff_array[t2i(idx, 1, 2, 3, DD)], sh_coeff_array[t2i(idx, 2, 2, 3, DD)]);
        float3 coeff3 = float3(sh_coeff_array[t2i(idx, 0, 3, 3, DD)], sh_coeff_array[t2i(idx, 1, 3, 3, DD)], sh_coeff_array[t2i(idx, 2, 3, 3, DD)]);
        float x = dir.x, y = dir.y, z = dir.z;

        rgb = rgb + coeff1 * SH_C1[0] * y + coeff2 * SH_C1[1] * z + coeff3 * SH_C1[0] * x;

        if (D > 4) {
            // SH2
            float3 coeff4 = float3(sh_coeff_array[t2i(idx, 0, 4, 3, DD)], sh_coeff_array[t2i(idx, 1, 4, 3, DD)], sh_coeff_array[t2i(idx, 2, 4, 3, DD)]);
            float3 coeff5 = float3(sh_coeff_array[t2i(idx, 0, 5, 3, DD)], sh_coeff_array[t2i(idx, 1, 5, 3, DD)], sh_coeff_array[t2i(idx, 2, 5, 3, DD)]);
            float3 coeff6 = float3(sh_coeff_array[t2i(idx, 0, 6, 3, DD)], sh_coeff_array[t2i(idx, 1, 6, 3, DD)], sh_coeff_array[t2i(idx, 2, 6, 3, DD)]);
            float3 coeff7 = float3(sh_coeff_array[t2i(idx, 0, 7, 3, DD)], sh_coeff_array[t2i(idx, 1, 7, 3, DD)], sh_coeff_array[t2i(idx, 2, 7, 3, DD)]);
            float3 coeff8 = float3(sh_coeff_array[t2i(idx, 0, 8, 3, DD)], sh_coeff_array[t2i(idx, 1, 8, 3, DD)], sh_coeff_array[t2i(idx, 2, 8, 3, DD)]);

            float x2 = x * x, y2 = y * y, z2 = z * z;
            float xy = x * y, yz = y * z, xz = x * z;

            rgb = rgb + coeff4 * SH_C2[0] * xy;
            rgb = rgb + coeff5 * SH_C2[1] * yz;
            rgb = rgb + coeff6 * SH_C2[2] * (2.0f * z2 - x2 - y2);
            rgb = rgb + coeff7 * SH_C2[1] * xz;
            rgb = rgb + coeff8 * SH_C2[0] * 0.5 * (x2 - y2);

            if (D > 9) {
                // SH3
                float3 coeff9 = float3(sh_coeff_array[t2i(idx, 0, 9, 3, DD)], sh_coeff_array[t2i(idx, 1, 9, 3, DD)], sh_coeff_array[t2i(idx, 2, 9, 3, DD)]);
                float3 coeff10 = float3(sh_coeff_array[t2i(idx, 0, 10, 3, DD)], sh_coeff_array[t2i(idx, 1, 10, 3, DD)], sh_coeff_array[t2i(idx, 2, 10, 3, DD)]);
                float3 coeff11 = float3(sh_coeff_array[t2i(idx, 0, 11, 3, DD)], sh_coeff_array[t2i(idx, 1, 11, 3, DD)], sh_coeff_array[t2i(idx, 2, 11, 3, DD)]);
                float3 coeff12 = float3(sh_coeff_array[t2i(idx, 0, 12, 3, DD)], sh_coeff_array[t2i(idx, 1, 12, 3, DD)], sh_coeff_array[t2i(idx, 2, 12, 3, DD)]);
                float3 coeff13 = float3(sh_coeff_array[t2i(idx, 0, 13, 3, DD)], sh_coeff_array[t2i(idx, 1, 13, 3, DD)], sh_coeff_array[t2i(idx, 2, 13, 3, DD)]);
                float3 coeff14 = float3(sh_coeff_array[t2i(idx, 0, 14, 3, DD)], sh_coeff_array[t2i(idx, 1, 14, 3, DD)], sh_coeff_array[t2i(idx, 2, 14, 3, DD)]);
                float3 coeff15 = float3(sh_coeff_array[t2i(idx, 0, 15, 3, DD)], sh_coeff_array[t2i(idx, 1, 15, 3, DD)], sh_coeff_array[t2i(idx, 2, 15, 3, DD)]);

                rgb = rgb + coeff9 * SH_C3[0] * y * (3.0f * x2 - y2);
                rgb = rgb + coeff10 * SH_C3[1] * xy * z;
                rgb = rgb + coeff11 * SH_C3[2] * y * (4.0f * z2 - x2 - y2);
                rgb = rgb + coeff12 * SH_C3[3] * z * (2.0f * z2 - 3.0f * x2 - 3.0f * y2);
                rgb = rgb + coeff13 * SH_C3[2] * x * (4.0f * z2 - x2 - y2);
                rgb = rgb + coeff14 * SH_C3[1] * z * 0.5 * (x2 - y2);
                rgb = rgb + coeff15 * SH_C3[0] * x * (x2 - 3.0f * y2);
            }
        }
    }
    */

    value_array[3 * idx + 0] = max(rgb.x + 0.5, 0.0);
    value_array[3 * idx + 1] = max(rgb.y + 0.5, 0.0);
    value_array[3 * idx + 2] = max(rgb.z + 0.5, 0.0);
}