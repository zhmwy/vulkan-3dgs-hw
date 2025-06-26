

void sort_small(inout RWStructuredBuffer<float> d, int n)
{
    int j;
    for (int i = 1; i < n; i++)
    {
        float tmp = d[i];
        for (j = i; j >= 1 && tmp < d[j-1]; j--)
        {
            d[j] = d[j-1];
        }
        d[j] = tmp;
    }
}

void Get3DPoint(RWStructuredBuffer<float> K, int2 p, float depth, out float3 X)
{
    // 从像素坐标和深度计算3D点（相机坐标系）
    X.x = depth * (p.x - K.z) / K.x;  // X = (u - cx) * depth / fx
    X.y = depth * (p.y - K.w) / K.y;  // Y = (v - cy) * depth / fy
    X.z = depth;                      // Z = depth
}