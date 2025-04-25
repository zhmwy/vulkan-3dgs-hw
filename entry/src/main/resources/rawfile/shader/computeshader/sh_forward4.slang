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
[[vk::binding(0)]] StructuredBuffer<float4> sh_coeff_array;
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

    if (visible_array[idx] == 0)
        return;


    float3 dir = {
        xyz_array[3 * idx + 0] - camera_center[0],
        xyz_array[3 * idx + 1] - camera_center[1],
        xyz_array[3 * idx + 2] - camera_center[2],
    };
    dir = dir / sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);

    // SH0
    float3 coeff0 = float3(sh_coeff_array[idx*3+0].x, sh_coeff_array[idx*3+1].x, sh_coeff_array[idx*3+2].x);
    float3 rgb = SH_C0 * coeff0;

    if (D > 1) {
        // SH1
        float3 coeff1 = float3(sh_coeff_array[idx*3+0].y, sh_coeff_array[idx*3+1].y, sh_coeff_array[idx*3+2].y);
        float3 coeff2 = float3(sh_coeff_array[idx*3+0].z, sh_coeff_array[idx*3+1].z, sh_coeff_array[idx*3+2].z);
        float3 coeff3 = float3(sh_coeff_array[idx*3+0].w, sh_coeff_array[idx*3+1].w, sh_coeff_array[idx*3+2].w);
        float x = dir.x, y = dir.y, z = dir.z;

        rgb = rgb + coeff1 * SH_C1[0] * y + coeff2 * SH_C1[1] * z + coeff3 * SH_C1[0] * x;
    }

    value_array[3 * idx + 0] = max(rgb.x + 0.5, 0.0);
    value_array[3 * idx + 1] = max(rgb.y + 0.5, 0.0);
    value_array[3 * idx + 2] = max(rgb.z + 0.5, 0.0);
}