#include "utils.slang"

[[vk::binding(0)]] RWStructuredBuffer<float> xyz_flow_params;
[[vk::binding(1)]] RWStructuredBuffer<float> rot_flow_params;

[[vk::binding(2)]] StructuredBuffer<float> dL_dxyz_flow;
[[vk::binding(3)]] StructuredBuffer<float> dL_drot_flow;

[[vk::binding(4)]] RWStructuredBuffer<float> xyz_flow_avg;
[[vk::binding(5)]] RWStructuredBuffer<float> rot_flow_avg;

[[vk::binding(6)]] RWStructuredBuffer<float> xyz_flow_exp_avg;
[[vk::binding(7)]] RWStructuredBuffer<float> rot_flow_exp_avg;

[[vk::binding(8)]] StructuredBuffer<int32_t> curPointsNum;
[[vk::binding(9)]] RWStructuredBuffer<uint32_t> visible;

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

// static const float extr_lr = 0.001;

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
    float _flow_steps;
};

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

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{

    // cast float to uint
    const int xyz_size = int(curPointsNum[0]);
    const uint width = uint(width);
    const uint height = uint(height);
    const uint flow_steps = uint(_flow_steps);

    uint idx = dispatchThreadID.x;
    if (idx >= xyz_size) return;

    if (visible[idx] == 0) return;  // skip invisible points

    // update xyz_flow_param
    update_adam(
        idx,
        18,
        dL_dxyz_flow,
        xyz_flow_params,
        xyz_flow_exp_avg,
        xyz_flow_avg,
        position_lr,
        _training_step - flow_steps
    );
   
    // update rotate_flow_param
    update_adam(
        idx,
        24,
        dL_drot_flow,
        rot_flow_params,
        rot_flow_exp_avg,
        rot_flow_avg,
        rotate_lr,
        _training_step - flow_steps
    );
}