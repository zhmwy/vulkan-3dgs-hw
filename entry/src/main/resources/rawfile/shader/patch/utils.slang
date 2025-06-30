

//float random(float2 st, uint iCall) {
//    return fract(sin(dot(st, float2(12.9898, 78.233))) * 43758.5453123 * iCall);
//}

// 移动端友好的Xorshift随机数
uint mobileRand(in uint2 coord, in uint frame) {
    uint seed = (coord.x * 1973u) ^ (coord.y * 9277u) + frame;
    seed ^= seed << 13u;
    seed ^= seed >> 17u;
    seed ^= seed << 5u;
    return seed;
}

float random(in uint2 coord, in uint frame) {
    uint2 offCoord = {197, 197};
    return (float(mobileRand(coord, frame)) * (1.0 / 4294967296.0) + float(mobileRand(coord+offCoord, frame)) * (1.0 / 4294967296.0)) / 2.0;
}

float3 Get3DPoint(float4 cameraK, int2 p, float depth)
{
    float3 X;
    X.x = depth * (p.x - cameraK.z) / cameraK.x;
    X.y = depth * (p.y - cameraK.w) / cameraK.y;
    X.z = depth;
    return X;
}

float4 GetViewDirection(float4 cameraK, const int2 p, const float depth)
{
    float3 X = Get3DPoint(cameraK, p, depth);
    float norm = sqrt(X[0] * X[0] + X[1] * X[1] + X[2] * X[2]);

    float4 view_direction;
    view_direction.x = X[0] / norm;
    view_direction.y = X[1] / norm;
    view_direction.z = X[2] / norm;
    view_direction.w = 0;
    return view_direction;
}

void NormalizeVec3(inout float4 vec)
{
    const float normSquared = vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
    const float inverse_sqrt = rsqrt(normSquared);
    vec.xyz *= inverse_sqrt;
}

float4 GenerateRandomNormal(float4 cameraK, const int2 p, const float depth)
{
    float q1 = 1.0f;
    float q2 = 1.0f;
    float s  = 2.0f;

    int idx = currentImage;
    while (s >= 1.0f) {
        q1 = 2.0f * random(p, idx) - 1.0f;
        q2 = 2.0f * random(p + int2(37, 17), idx + 1) - 1.0f;
        s = q1 * q1 + q2 * q2;
        idx = idx + 1;
    }

    idx = (uint)(random(p + int2(113, 127), currentImage)) * 32771;

    const float sq = sqrt(1.0f - s);
    float4 normal;
    if (idx % 3 == 0) {
        normal.x = 2.0f * q1 * sq;
        normal.y = 2.0f * q2 * sq;
        normal.z = 1.0f - 2.0f * s;
    }
    else if (idx % 3 == 1) {
        normal.y = 2.0f * q1 * sq;
        normal.z = 2.0f * q2 * sq;
        normal.x = 1.0f - 2.0f * s;
    }
    else if (idx % 3 == 2) {
        normal.z = 2.0f * q1 * sq;
        normal.x = 2.0f * q2 * sq;
        normal.y = 1.0f - 2.0f * s;
    }
    normal.w = 0.0f;

    float4 view_direction = GetViewDirection(cameraK, p, depth);
    float dot_product = normal.x * view_direction.x + normal.y * view_direction.y + normal.z * view_direction.z;
    if (dot_product > 0.0f) {
        normal.x = -normal.x;
        normal.y = -normal.y;
        normal.z = - normal.z;
    }
    NormalizeVec3(normal);
    return normal;
}

float GetDistance2Origin(float4 cameraK, int2 p, const float depth, const float4 normal)
{
    float3 X = Get3DPoint(cameraK, p, depth);
    return -(normal.x * X.x + normal.y * X.y + normal.z * X.z);
}

// Define camera structure to match your CUDA code
struct Camera {
    float R[9];  // Rotation matrix (3x3 in row-major order)
    float t[3];  // Translation vector
    float K[4];  // Intrinsic matrix parameters (fx, fy, cx, cy)
};

// Compute homography function in Slang
void ComputeHomography(Camera ref_camera, Camera src_camera, float4 plane_hypothesis, inout float H[9]) {
    // Compute camera centers
    float3 ref_C = float3(
        -(ref_camera.R[0] * ref_camera.t[0] + ref_camera.R[3] * ref_camera.t[1] + ref_camera.R[6] * ref_camera.t[2]),
        -(ref_camera.R[1] * ref_camera.t[0] + ref_camera.R[4] * ref_camera.t[1] + ref_camera.R[7] * ref_camera.t[2]),
        -(ref_camera.R[2] * ref_camera.t[0] + ref_camera.R[5] * ref_camera.t[1] + ref_camera.R[8] * ref_camera.t[2])
    );

    float3 src_C = float3(
        -(src_camera.R[0] * src_camera.t[0] + src_camera.R[3] * src_camera.t[1] + src_camera.R[6] * src_camera.t[2]),
        -(src_camera.R[1] * src_camera.t[0] + src_camera.R[4] * src_camera.t[1] + src_camera.R[7] * src_camera.t[2]),
        -(src_camera.R[2] * src_camera.t[0] + src_camera.R[5] * src_camera.t[1] + src_camera.R[8] * src_camera.t[2])
    );

    // Compute relative rotation
    float R_relative[9];
    R_relative[0] = src_camera.R[0] * ref_camera.R[0] + src_camera.R[1] * ref_camera.R[1] + src_camera.R[2] * ref_camera.R[2];
    R_relative[1] = src_camera.R[0] * ref_camera.R[3] + src_camera.R[1] * ref_camera.R[4] + src_camera.R[2] * ref_camera.R[5];
    R_relative[2] = src_camera.R[0] * ref_camera.R[6] + src_camera.R[1] * ref_camera.R[7] + src_camera.R[2] * ref_camera.R[8];
    R_relative[3] = src_camera.R[3] * ref_camera.R[0] + src_camera.R[4] * ref_camera.R[1] + src_camera.R[5] * ref_camera.R[2];
    R_relative[4] = src_camera.R[3] * ref_camera.R[3] + src_camera.R[4] * ref_camera.R[4] + src_camera.R[5] * ref_camera.R[5];
    R_relative[5] = src_camera.R[3] * ref_camera.R[6] + src_camera.R[4] * ref_camera.R[7] + src_camera.R[5] * ref_camera.R[8];
    R_relative[6] = src_camera.R[6] * ref_camera.R[0] + src_camera.R[7] * ref_camera.R[1] + src_camera.R[8] * ref_camera.R[2];
    R_relative[7] = src_camera.R[6] * ref_camera.R[3] + src_camera.R[7] * ref_camera.R[4] + src_camera.R[8] * ref_camera.R[5];
    R_relative[8] = src_camera.R[6] * ref_camera.R[6] + src_camera.R[7] * ref_camera.R[7] + src_camera.R[8] * ref_camera.R[8];

    // Compute relative translation
    float3 C_relative = ref_C - src_C;
    float3 t_relative = float3(
        src_camera.R[0] * C_relative.x + src_camera.R[1] * C_relative.y + src_camera.R[2] * C_relative.z,
        src_camera.R[3] * C_relative.x + src_camera.R[4] * C_relative.y + src_camera.R[5] * C_relative.z,
        src_camera.R[6] * C_relative.x + src_camera.R[7] * C_relative.y + src_camera.R[8] * C_relative.z
    );

    // Compute homography components
    float n = plane_hypothesis.w;  // plane normal scale factor
    H[0] = R_relative[0] - t_relative.x * plane_hypothesis.x / n;
    H[1] = R_relative[1] - t_relative.x * plane_hypothesis.y / n;
    H[2] = R_relative[2] - t_relative.x * plane_hypothesis.z / n;
    H[3] = R_relative[3] - t_relative.y * plane_hypothesis.x / n;
    H[4] = R_relative[4] - t_relative.y * plane_hypothesis.y / n;
    H[5] = R_relative[5] - t_relative.y * plane_hypothesis.z / n;
    H[6] = R_relative[6] - t_relative.z * plane_hypothesis.x / n;
    H[7] = R_relative[7] - t_relative.z * plane_hypothesis.y / n;
    H[8] = R_relative[8] - t_relative.z * plane_hypothesis.z / n;

    // Apply intrinsic matrices
    float tmp[9];
    tmp[0] = H[0] / ref_camera.K[0];
    tmp[1] = H[1] / ref_camera.K[1];
    tmp[2] = -H[0] * ref_camera.K[2] / ref_camera.K[0] - H[1] * ref_camera.K[3] / ref_camera.K[1] + H[2];
    tmp[3] = H[3] / ref_camera.K[0];
    tmp[4] = H[4] / ref_camera.K[1];
    tmp[5] = -H[3] * ref_camera.K[2] / ref_camera.K[0] - H[4] * ref_camera.K[3] / ref_camera.K[1] + H[5];
    tmp[6] = H[6] / ref_camera.K[0];
    tmp[7] = H[7] / ref_camera.K[1];
    tmp[8] = -H[6] * ref_camera.K[2] / ref_camera.K[0] - H[7] * ref_camera.K[3] / ref_camera.K[1] + H[8];

    H[0] = src_camera.K[0] * tmp[0] + src_camera.K[2] * tmp[6];
    H[1] = src_camera.K[0] * tmp[1] + src_camera.K[2] * tmp[7];
    H[2] = src_camera.K[0] * tmp[2] + src_camera.K[2] * tmp[8];
    H[3] = src_camera.K[1] * tmp[3] + src_camera.K[3] * tmp[6];
    H[4] = src_camera.K[1] * tmp[4] + src_camera.K[3] * tmp[7];
    H[5] = src_camera.K[1] * tmp[5] + src_camera.K[3] * tmp[8];
    H[6] = tmp[6];
    H[7] = tmp[7];
    H[8] = tmp[8];
}