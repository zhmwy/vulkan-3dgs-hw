

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