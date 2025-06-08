[[vk::binding(0)]] RWStructuredBuffer<float> psnr_sum;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID) {
    uint32_t idx = dispatchThreadID.x;
    if (idx > 0)
        return;
    psnr_sum[0] = 0.0;
    psnr_sum[1] = 0.0;
    psnr_sum[2] = 0.0;
}
