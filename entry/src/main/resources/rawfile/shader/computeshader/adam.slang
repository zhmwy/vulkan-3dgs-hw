[[vk::binding(0)]] RWStructuredBuffer<float> xyz_array;
[[vk::binding(1)]] RWStructuredBuffer<float> scale_array;
[[vk::binding(2)]] RWStructuredBuffer<float4> rotate_array;
[[vk::binding(3)]] RWStructuredBuffer<float> feature_array;
[[vk::binding(4)]] RWStructuredBuffer<float> opacity_array;

[[vk::binding(5)]] StructuredBuffer<float> dL_dxyz_array;
[[vk::binding(6)]] StructuredBuffer<float> dL_dscale_array;
[[vk::binding(7)]] StructuredBuffer<float4> dL_drotate_array;
[[vk::binding(8)]] StructuredBuffer<float> dL_dfeature_array;
[[vk::binding(9)]] StructuredBuffer<float> dL_dopacity_array;

[[vk::binding(10)]] RWStructuredBuffer<float> xyz_array_exp_avg;
[[vk::binding(11)]] RWStructuredBuffer<float> scale_array_exp_avg;
[[vk::binding(12)]] RWStructuredBuffer<float4> rotate_array_exp_avg;
[[vk::binding(13)]] RWStructuredBuffer<float> feature_array_exp_avg;
[[vk::binding(14)]] RWStructuredBuffer<float> opacity_array_exp_avg;

[[vk::binding(15)]] RWStructuredBuffer<float> xyz_array_avg;
[[vk::binding(16)]] RWStructuredBuffer<float> scale_array_avg;
[[vk::binding(17)]] RWStructuredBuffer<float4> rotate_array_avg;
[[vk::binding(18)]] RWStructuredBuffer<float> feature_array_avg;
[[vk::binding(19)]] RWStructuredBuffer<float> opacity_array_avg;

[[vk::binding(20)]] StructuredBuffer<int32_t> curPointsNum;
[[vk::binding(21)]] RWStructuredBuffer<uint32_t> visible;

// static const int D = 16;

// Adam hyperparameters
static const float beta1 = 0.9;          // Decay rate for the first moment
static const float beta2 = 0.999; // Decay rate for the second moment
static const float eps = 1e-8;    // Small constant to avoid division by zero
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
    float current_img_idx;
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

        // Compute the step size using Adam formula
        float step = -learning_rate * m / (sqrt(v) + eps);

        // Update the parameter
        param_array[offset] += step;

        // Write back the updated first-order and second-order moments
        m_array[offset] = m;
        v_array[offset] = v;
    }
}

void update_adam_for_features(
    uint idx,                              // Index of the current element to process
    uint vec_length,                       // Length of the vector (e.g., 3, 48, etc.)
    StructuredBuffer<float> grad_array,    // Input gradient array (read-only)
    RWStructuredBuffer<float> param_array, // Input parameter array to be updated (read-write)
    RWStructuredBuffer<float> m_array,     // First-order moment array (read-write)
    RWStructuredBuffer<float> v_array,     // Second-order moment array (read-write)
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

        // Compute the step size using Adam formula
        float step;
        if (i % D == 0)
            step = -learning_rate_features * m / (sqrt(v) + eps);
        else 
            step = -learning_rate_features_rest * m / (sqrt(v) + eps);

        // Update the parameter
        param_array[offset] += step;

        // Write back the updated first-order and second-order moments
        m_array[offset] = m;
        v_array[offset] = v;
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
        position_lr
    );

    // update scale_array
    update_adam(
        idx,
        3,
        dL_dscale_array,
        scale_array,
        scale_array_exp_avg,
        scale_array_avg,
        scale_lr
    );

    // update rotate_array
    float4 grad_rotate = dL_drotate_array[idx];
    float4 m_rotate = rotate_array_exp_avg[idx];
    float4 v_rotate = rotate_array_avg[idx];

    m_rotate = beta1 * m_rotate + (1.0 - beta1) * grad_rotate;
    v_rotate = beta2 * v_rotate + (1.0 - beta2) * grad_rotate * grad_rotate;

    float4 step_rotate = -rotate_lr * m_rotate / (sqrt(v_rotate) + eps);
    rotate_array[idx] += step_rotate;

    rotate_array_exp_avg[idx] = m_rotate;
    rotate_array_avg[idx] = v_rotate;

    update_adam_for_features(
        idx,
        ((int)D) * 3,
        dL_dfeature_array,
        feature_array,
        feature_array_exp_avg,
        feature_array_avg
    );

    // update opacity_array
    update_adam(
        idx,
        1,
        dL_dopacity_array,
        opacity_array,
        opacity_array_exp_avg,
        opacity_array_avg,
        opacity_lr
    );
}