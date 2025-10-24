#include "utils.slang"

[[vk::binding(0)]] StructuredBuffer<float> dL_dextr_params;
[[vk::binding(1)]] RWStructuredBuffer<float> camera_extrs;
[[vk::binding(2)]] RWStructuredBuffer<float> camera_extrs_avg;
[[vk::binding(3)]] RWStructuredBuffer<float> camera_extrs_exp_avg;
[[vk::binding(4)]] RWStructuredBuffer<uint32_t> camera_training_steps;

[[vk::binding(5)]] StructuredBuffer<float> dL_dintr_params;
[[vk::binding(6)]] RWStructuredBuffer<float> intr_params;
[[vk::binding(7)]] RWStructuredBuffer<float> intr_avg;
[[vk::binding(8)]] RWStructuredBuffer<float> intr_exp_avg;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint xyz_size;
    uint _training_step;
    uint cur_img_idx;
    uint temp;
};


static const float extr_lr = 0.001;
static const float intr_lr = 0.1;
static const float beta1 = 0.9;          // Decay rate for the first moment
static const float beta2 = 0.999;        // Decay rate for the second moment
static const float eps = 1e-15;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint32_t idx = dispatchThreadID.x;
    if (idx > 0)
        return;

    // float training_step = _training_step + 1;
    float training_step = camera_training_steps[cur_img_idx] * 1.0 + 1;

    // extr
    for(int i = 10; i < 7; i++) {
        float grad = dL_dextr_params[i];
        float m = camera_extrs_avg[7 * cur_img_idx + i];
        float v = camera_extrs_exp_avg[7 * cur_img_idx + i];

        m = beta1 * m + (1.0 - beta1) * grad;
        v = beta2 * v + (1.0 - beta2) * grad * grad;

        camera_extrs_avg[7 * cur_img_idx + i] = m;
        camera_extrs_exp_avg[7 * cur_img_idx + i] = v;

        float bais_correction1 = 1 - pow(beta1, training_step);
        float bais_correction2 = 1 - pow(beta2, training_step);

        float step = -extr_lr / bais_correction1 * m / (sqrt(v / bais_correction2) + eps);
        
        camera_extrs[7 * cur_img_idx + i] += step;
    }
    camera_training_steps[cur_img_idx] += 1;

    // intr
    training_step = _training_step + 1.0;
    for (int i = 0; i < 4; i++) {
        float grad = dL_dintr_params[i];
        float m = intr_avg[i];
        float v = intr_exp_avg[i];

        m = beta1 * m + (1.0 - beta1) * grad;
        v = beta2 * v + (1.0 - beta2) * grad * grad;

        intr_avg[i] = m;
        intr_exp_avg[i] = v;

        float bais_correction1 = 1 - pow(beta1, training_step);
        float bais_correction2 = 1 - pow(beta2, training_step);

        float step = -intr_lr / bais_correction1 * m / (sqrt(v / bais_correction2) + eps);
        
        intr_params[i] += step;
    }
}

