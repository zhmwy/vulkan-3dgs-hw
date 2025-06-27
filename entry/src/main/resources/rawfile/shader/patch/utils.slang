

// 高质量随机数生成器 (输入为归一化坐标)
// 输入: normCoord - 归一化像素坐标 (范围[0,1])
//       callCount - 调用次数/帧计数
// 输出: [0,1)范围内的随机数
float randomNormalized(float2 normCoord, uint callCount)
{
    // 将归一化坐标映射到大整数范围
    uint2 intCoords = uint2(
        uint(normCoord.x * 4294967295.0),  // 将[0,1]映射到32位整数范围
        uint(normCoord.y * 4294967295.0)
    );

    // PCG哈希算法 (改良版)
    uint seed = (intCoords.x * 1973) ^ (intCoords.y * 9277) + callCount * 26699;
    seed = seed * 747796405 + 1;
    seed = (seed ^ (seed >> 16)) * 277803737;
    seed = seed ^ (seed >> 16);

    // 转换为[0,1)范围的高精度浮点数
    return float(seed) * (1.0 / 4294967296.0);
}

// 基于像素位置和调用次数的伪随机数生成器
// 返回值范围: [0.0, 1.0)
float random(uint2 pixelCoord, uint callCount)
{
    // 使用哈希算法混合坐标和调用次数
    uint seed = (pixelCoord.x * 1973) ^ (pixelCoord.y * 9277) + callCount * 26699;

    // 简单的PCG哈希算法变种
    seed = seed * 747796405 + 1;
    uint word = (seed >> ((seed >> 28) + 4)) ^ seed;
    uint result = (word * 277803737) >> 22;

    // 转换为[0,1)范围的浮点数
    return fract(float(result) / 4294967296.0);
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
    int idx = 1;

    // 使用帧索引和像素坐标作为随机种子
    while (s >= 1.0f) {
        q1 = 2.0f * random(p, idx) - 1.0f;
        q2 = 2.0f * random(p + int2(37, 17), idx + 1) - 1.0f;
        s = q1 * q1 + q2 * q2;
        idx = idx + 1;
    }

    const float sq = sqrt(1.0f - s);
    float4 normal;
    normal.x = 2.0f * q1 * sq;
    normal.y = 2.0f * q2 * sq;
    normal.z = 1.0f - 2.0f * s;
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