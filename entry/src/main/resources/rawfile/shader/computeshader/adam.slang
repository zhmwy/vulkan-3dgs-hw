#include "utils.slang"

[[vk::binding(0)]] RWStructuredBuffer<float> xyz_array;
[[vk::binding(1)]] RWStructuredBuffer<float> scale_array;
[[vk::binding(2)]] RWStructuredBuffer<float> rotate_array;
[[vk::binding(3)]] RWStructuredBuffer<float> feature_array;
[[vk::binding(4)]] RWStructuredBuffer<float> opacity_array;

[[vk::binding(5)]] StructuredBuffer<float> dL_dxyz_array;
[[vk::binding(6)]] StructuredBuffer<float> dL_dscale_array;
[[vk::binding(7)]] StructuredBuffer<float> dL_drotate_array;
[[vk::binding(8)]] StructuredBuffer<float> dL_dfeature_array;
[[vk::binding(9)]] StructuredBuffer<float> dL_dopacity_array;

[[vk::binding(10)]] RWStructuredBuffer<float> xyz_array_exp_avg;
[[vk::binding(11)]] RWStructuredBuffer<float> scale_array_exp_avg;
[[vk::binding(12)]] RWStructuredBuffer<float> rotate_array_exp_avg;
[[vk::binding(13)]] RWStructuredBuffer<float> feature_array_exp_avg;
[[vk::binding(14)]] RWStructuredBuffer<float> opacity_array_exp_avg;

[[vk::binding(15)]] RWStructuredBuffer<float> xyz_array_avg;
[[vk::binding(16)]] RWStructuredBuffer<float> scale_array_avg;
[[vk::binding(17)]] RWStructuredBuffer<float> rotate_array_avg;
[[vk::binding(18)]] RWStructuredBuffer<float> feature_array_avg;
[[vk::binding(19)]] RWStructuredBuffer<float> opacity_array_avg;

[[vk::binding(20)]] StructuredBuffer<int32_t> curPointsNum;
[[vk::binding(21)]] RWStructuredBuffer<uint32_t> visible;

// [[vk::binding(22)]] RWStructuredBuffer<float> dL_dextr_params;
// [[vk::binding(23)]] RWStructuredBuffer<float> camera_extrs;
// [[vk::binding(24)]] RWStructuredBuffer<float> camera_extrs_avg;
// [[vk::binding(25)]] RWStructuredBuffer<float> camera_extrs_exp_avg;

// static const int D = 16;

// Adam hyperparameters
static const float beta1 = 0.9;          // Decay rate for the first moment
static const float beta2 = 0.999; // Decay rate for the second moment
static const float eps = 1e-15;    // Small constant to avoid division by zero
// const float position_lr = 0.00016; // Learning rate
static const float scale_lr = 0.005;           // Learning rate
static const float rotate_lr = 0.001;   // Learning rate
static const float opacity_lr = 0.05;  // Learning rate
static const float learning_rate_features = 0.0025;
static const float learning_rate_features_rest = 0.0025 / 20;

static const float extr_lr = 0.001;

[[vk::push_constant]]
cbuffer PushConstants
{
    float xyz_size;
    float width;
    float height;
    float position_lr;
    float D;
    float extent;
    float _training_step;
};

// Function to perform Adam optimizer updates for float3 data stored in separate float arrays
void update_adam_float3(
    uint idx,                              // Index of the current element to process
    StructuredBuffer<float> grad_array,    // Input gradient array (read-only)
    RWStructuredBuffer<float> param_array, // Input parameter array to be updated (read-write)
    RWStructuredBuffer<float> m_array,     // First-order moment array (read-write)
    RWStructuredBuffer<float> v_array,     // Second-order moment array (read-write)
    float learning_rate,                   // Learning rate for Adam optimizer
)
{
    // Read the gradient for the current element
    float3 grad = float3(
        grad_array[3 * idx],
        grad_array[3 * idx + 1],
        grad_array[3 * idx + 2]
    );

    // Read the first-order moment for the current element
    float3 m = float3(
        m_array[3 * idx],
        m_array[3 * idx + 1],
        m_array[3 * idx + 2]
    );

    // Read the second-order moment for the current element
    float3 v = float3(
        v_array[3 * idx],
        v_array[3 * idx + 1],
        v_array[3 * idx + 2]
    );

    // Update first-order moment (moving average of gradients)
    m = beta1 * m + (1.0 - beta1) * grad;

    // Update second-order moment (moving average of squared gradients)
    v = beta2 * v + (1.0 - beta2) * grad * grad;

    // Compute the step size using Adam formula
    float3 step = -learning_rate * m / (sqrt(v) + eps);

    // Update the parameter array with the computed step
    param_array[3 * idx] += step.x;
    param_array[3 * idx + 1] += step.y;
    param_array[3 * idx + 2] += step.z;

    // Write back the updated first-order moment
    m_array[3 * idx] = m.x;
    m_array[3 * idx + 1] = m.y;
    m_array[3 * idx + 2] = m.z;

    // Write back the updated second-order moment
    v_array[3 * idx] = v.x;
    v_array[3 * idx + 1] = v.y;
    v_array[3 * idx + 2] = v.z;
}

void update_adam(
    uint idx,                              // Index of the current element to process
    uint vec_length,                       // Length of the vector (e.g., 3, 48, etc.)
    StructuredBuffer<float> grad_array,    // Input gradient array (read-only)
    RWStructuredBuffer<float> param_array, // Input parameter array to be updated (read-write)
    RWStructuredBuffer<float> m_array,     // First-order moment array (read-write)
    RWStructuredBuffer<float> v_array,     // Second-order moment array (read-write)
    float learning_rate,                   // Learning rate for Adam optimizer
    float training_step
)
{

    // Loop through each dimension of the vector
    for (uint i = 0; i < vec_length; ++i) {
        uint offset = idx * vec_length + i;

        // Read the gradient, first-order moment, and second-order moment for the current dimension
        float grad = grad_array[offset];
        float m = m_array[offset];
        float v = v_array[offset];

        // Update first-order moment (moving average of gradients)
        m = beta1 * m + (1.0 - beta1) * grad;

        // Update second-order moment (moving average of squared gradients)
        v = beta2 * v + (1.0 - beta2) * grad * grad;
        
        m_array[offset] = m;
        v_array[offset] = v;

        float bais_correction1 = 1 - pow(beta1, training_step);
        float bais_correction2 = 1 - pow(beta2, training_step);

        float step = -learning_rate / bais_correction1 * m / (sqrt(v / bais_correction2) + eps);

        // Compute the step size using Adam formula
        // float step = -learning_rate * m / (sqrt(v) + eps);

        // Update the parameter
        param_array[offset] += step;

        // Write back the updated first-order and second-order moments

    }
}

void update_adam_for_features(
    uint idx,                              // Index of the current element to process
    uint vec_length,                       // Length of the vector (e.g., 3, 48, etc.)
    StructuredBuffer<float> grad_array,    // Input gradient array (read-only)
    RWStructuredBuffer<float> param_array, // Input parameter array to be updated (read-write)
    RWStructuredBuffer<float> m_array,     // First-order moment array (read-write)
    RWStructuredBuffer<float> v_array,     // Second-order moment array (read-write)
    float training_step
)
{
    for (uint i = 0; i < 3; i++) {
        for (uint j = 0; j < vec_length; j++) {
            uint offset = idx * DD * 3 + i * DD + j;
            float lr = 0.0;
            if (j == 0)
                lr = learning_rate_features;
            else
                lr = learning_rate_features_rest;

            // Read the gradient, first-order moment, and second-order moment for the current dimension
            float grad = grad_array[offset];
            float m = m_array[offset];
            float v = v_array[offset];

            // Update first-order moment (moving average of gradients)
            m = beta1 * m + (1.0 - beta1) * grad;

            // Update second-order moment (moving average of squared gradients)
            v = beta2 * v + (1.0 - beta2) * grad * grad;

            // Write back the updated first-order and second-order moments
            m_array[offset] = m;
            v_array[offset] = v;

            float bais_correction1 = 1 - pow(beta1, training_step);
            float bais_correction2 = 1 - pow(beta2, training_step);

            // Compute the step size using Adam formula
            float step = -lr / bais_correction1 * m / (sqrt(v / bais_correction2) + eps);

            // Update the parameter
            param_array[offset] += step;
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{

    // cast float to uint
    const int xyz_size = int(curPointsNum[0]);
    const uint width = uint(width);
    const uint height = uint(height);

    uint idx = dispatchThreadID.x;
    if (idx >= xyz_size) return;

    if (visible[idx] == 0) return; // skip invisible points

    // update xyz_array
    update_adam(
        idx,
        3,
        dL_dxyz_array,
        xyz_array,
        xyz_array_exp_avg,
        xyz_array_avg,
        position_lr,
        _training_step + 1
    );

    // update scale_array
    update_adam(
        idx,
        3,
        dL_dscale_array,
        scale_array,
        scale_array_exp_avg,
        scale_array_avg,
        scale_lr,
        _training_step + 1
    );

    update_adam(
        idx, 4,
        dL_drotate_array,
        rotate_array,
        rotate_array_exp_avg,
        rotate_array_avg,
        rotate_lr,
        _training_step + 1
    );

    update_adam_for_features(
        idx,
        (int)D,
        // 48, 
        dL_dfeature_array,
        feature_array,
        feature_array_exp_avg,
        feature_array_avg,
        _training_step + 1
    );

    // update opacity_array
    update_adam(
        idx,
        1,
        dL_dopacity_array,
        opacity_array,
        opacity_array_exp_avg,
        opacity_array_avg,
        opacity_lr,
        _training_step + 1
    );
}