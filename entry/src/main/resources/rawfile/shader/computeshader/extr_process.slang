[[vk::binding(0)]] StructuredBuffer<float> extr_params;
[[vk::binding(1)]] RWStructuredBuffer<float> extr_act;
[[vk::binding(2)]] RWStructuredBuffer<float> extr;
[[vk::binding(3)]] RWStructuredBuffer<float> camera_center;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint xyz_size;
    uint width;
    uint height;
    uint temp;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint32_t idx = dispatchThreadID.x;
    if (idx > 0)
        return;
    float w = extr_params[0];
    float x = extr_params[1];
    float y = extr_params[2];
    float z = extr_params[3];

    float norm_extr = sqrt(w * w + x * x + y * y + z * z);
    w = w / norm_extr;
    x = x / norm_extr;
    y = y / norm_extr;
    z = z / norm_extr;

    extr[0] = 1 - 2 * (y * y + z * z);
    extr[1] = 2 * (x * y - w * z);
    extr[2] = 2 * (x * z + w * y);
    extr[3] = extr_params[4];

    extr[4] = 2 * (x * y + w * z);
    extr[5] = 1 - 2 * (x * x + z * z);
    extr[6] = 2 * (y * z - w * x);
    extr[7] = extr_params[5];

    extr[8] = 2 * (x * z - w * y);
    extr[9] = 2 * (y * z + w * x);
    extr[10] = 1 - 2 * (x * x + y * y);
    extr[11] = extr_params[6];

    extr_act[0] = w;
    extr_act[1] = x;
    extr_act[2] = y;
    extr_act[3] = z;
    extr_act[4] = norm_extr;
    extr_act[5] = pow(norm_extr, 3);

    camera_center[0] = -(extr[0] * extr[3] + extr[4] * extr[7] + extr[8] * extr[11]);
    camera_center[1] = -(extr[1] * extr[3] + extr[5] * extr[7] + extr[9] * extr[11]);
    camera_center[2] = -(extr[2] * extr[3] + extr[6] * extr[7] + extr[10] * extr[11]);
}


