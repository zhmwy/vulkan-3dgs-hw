#include "utils.slang"

#define G_00 0.001028380123898387f
#define G_01 0.0075987582094967365f
#define G_02 0.036000773310661316f
#define G_03 0.10936068743467331f
#define G_04 0.21300552785396576f
#define G_05 0.26601171493530273f
#define G_06 0.21300552785396576f
#define G_07 0.10936068743467331f
#define G_08 0.036000773310661316f
#define G_09 0.0075987582094967365f
#define G_10 0.001028380123898387f

// block size
#define BX 16
#define BY 16

// shared memory size
#define SX (BX + 10)
#define SSX (BX + 10)
#define SY (BY + 10)

// convolution scratchpad size
#define CX (BX)
#define CCX (BX + 0)
#define CY (BY + 10)

groupshared float buf1[SY][SSX];
groupshared float buf2[SY][SSX];
groupshared float buf3[CY][CCX];

float get_pix_value(
    StructuredBuffer<float> img,
    int c, int y, int x, int H, int W
)
{
    if (x >= W || y >= H || x < 0 || y < 0) {
        return 0.0f;
    }
    else {
        return img[t2i(c, y, x, H, W)];
        // return img[t2i(c, y, x, H, W)];
    }
}

inline void load_into_shared(
    out float pixels[SY][SSX],
    StructuredBuffer<float> img,
    int H, int W, int c, int item_idx,
    int block_idx_x, int block_idx_y, int pix_x, int pix_y
)
{
    int start_y = block_idx_y * BY;
    int start_x = block_idx_x * BX;

    const int cnt = SY * SX;
    const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);


    for (int b = 0; b < num_blocks; b++) {
        int tid = b * (BX * BY) + item_idx;
        if (tid < cnt) {
            int local_y = tid / SX;
            int local_x = tid % SX;
            int y = start_y + local_y;
            int x = start_x + local_x;
            float one = get_pix_value(img, c, y - 5, x - 5, H, W);
            pixels[local_y][local_x] = one;
        }
    }
}

inline void load_into_shared_buf1(
    StructuredBuffer<float> img,
    int H, int W, int c, int item_idx,
    int block_idx_x, int block_idx_y, int pix_x, int pix_y
)
{
    int start_y = block_idx_y * BY;
    int start_x = block_idx_x * BX;

    const int cnt = SY * SX;
    const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);

    for (int b = 0; b < num_blocks; b++) {
        int tid = b * (BX * BY) + item_idx;
        if (tid < cnt) {
            int local_y = tid / SX;
            int local_x = tid % SX;
            int y = start_y + local_y;
            int x = start_x + local_x;
            float one = get_pix_value(img, c, y - 5, x - 5, H, W);
            buf1[local_y][local_x] = one;
        }
    }
}

inline void load_into_shared_buf2(
    StructuredBuffer<float> img,
    int H, int W, int c, int item_idx,
    int block_idx_x, int block_idx_y, int pix_x, int pix_y
)
{
    int start_y = block_idx_y * BY;
    int start_x = block_idx_x * BX;

    const int cnt = SY * SX;
    const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);

    for (int b = 0; b < num_blocks; b++) {
        int tid = b * (BX * BY) + item_idx;
        if (tid < cnt) {
            int local_y = tid / SX;
            int local_x = tid % SX;
            int y = start_y + local_y;
            int x = start_x + local_x;
            float one = get_pix_value(img, c, y - 5, x - 5, H, W);
            buf2[local_y][local_x] = one;
        }
    }
}

inline void flush_conv_scratch(
    out float buf[CY][CCX], 
    int item_idx
)
{
    const int cnt = CY * CX;
    const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
    for (int b = 0; b < num_blocks; ++b) {
        const int tid = b * (BX * BY) + item_idx;
        if (tid < cnt) {
            const int local_y = tid / CX;
            const int local_x = tid % CX;
            buf[local_y][local_x] = 0.0f;
        }
    }
}

inline void flush_conv_scratch_buf3(
    int item_idx
)
{
    const int cnt = CY * CX;
    const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
    for (int b = 0; b < num_blocks; ++b) {
        const int tid = b * (BX * BY) + item_idx;
        if (tid < cnt) {
            const int local_y = tid / CX;
            const int local_x = tid % CX;
            buf3[local_y][local_x] = 0.0f;
        }
    }
}

inline void multiply_shared_mem_buf1_buf2(
    int item_idx
) {
    const int cnt = SY * SX;
    const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
    for (int b = 0; b < num_blocks; ++b) {
        int tid = b * (BX * BY) + item_idx;
        if (tid < cnt) {
            int local_y = tid / SX;
            int local_x = tid % SX;
            float one = buf1[local_y][local_x];
            float two = buf2[local_y][local_x];
            buf1[local_y][local_x] = one * two;
        }
    }
}

inline float do_sq(float val) {
    return val * val;
}
inline void do_separable_conv_x_buf2(
    int groupThreadID_x,
    int groupThreadID_y,
    bool sq = false
)
{
    int local_y = groupThreadID_y;
    int local_x = groupThreadID_x + 5;
    float val = 0.0f;

    if (sq) {
        val += G_00 * do_sq(buf2[local_y][local_x - 5]);
        val += G_01 * do_sq(buf2[local_y][local_x - 4]);
        val += G_02 * do_sq(buf2[local_y][local_x - 3]);
        val += G_03 * do_sq(buf2[local_y][local_x - 2]);
        val += G_04 * do_sq(buf2[local_y][local_x - 1]);
        val += G_05 * do_sq(buf2[local_y][local_x]);
        val += G_06 * do_sq(buf2[local_y][local_x + 1]);
        val += G_07 * do_sq(buf2[local_y][local_x + 2]);
        val += G_08 * do_sq(buf2[local_y][local_x + 3]);
        val += G_09 * do_sq(buf2[local_y][local_x + 4]);
        val += G_10 * do_sq(buf2[local_y][local_x + 5]);
    }
    else {
        val += G_00 * buf2[local_y][local_x - 5];
        val += G_01 * buf2[local_y][local_x - 4];
        val += G_02 * buf2[local_y][local_x - 3];
        val += G_03 * buf2[local_y][local_x - 2];
        val += G_04 * buf2[local_y][local_x - 1];
        val += G_05 * buf2[local_y][local_x];
        val += G_06 * buf2[local_y][local_x + 1];
        val += G_07 * buf2[local_y][local_x + 2];
        val += G_08 * buf2[local_y][local_x + 3];
        val += G_09 * buf2[local_y][local_x + 4];
        val += G_10 * buf2[local_y][local_x + 5];
    }
    buf3[local_y][local_x - 5] = val;

    val = 0.0;
    local_y = groupThreadID_y + BY;
    if (local_y < SY) {
        if (sq) {
            val += G_00 * do_sq(buf2[local_y][local_x - 5]);
            val += G_01 * do_sq(buf2[local_y][local_x - 4]);
            val += G_02 * do_sq(buf2[local_y][local_x - 3]);
            val += G_03 * do_sq(buf2[local_y][local_x - 2]);
            val += G_04 * do_sq(buf2[local_y][local_x - 1]);
            val += G_05 * do_sq(buf2[local_y][local_x]);
            val += G_06 * do_sq(buf2[local_y][local_x + 1]);
            val += G_07 * do_sq(buf2[local_y][local_x + 2]);
            val += G_08 * do_sq(buf2[local_y][local_x + 3]);
            val += G_09 * do_sq(buf2[local_y][local_x + 4]);
            val += G_10 * do_sq(buf2[local_y][local_x + 5]);
        }
        else {
            val += G_00 * buf2[local_y][local_x - 5];
            val += G_01 * buf2[local_y][local_x - 4];
            val += G_02 * buf2[local_y][local_x - 3];
            val += G_03 * buf2[local_y][local_x - 2];
            val += G_04 * buf2[local_y][local_x - 1];
            val += G_05 * buf2[local_y][local_x];
            val += G_06 * buf2[local_y][local_x + 1];
            val += G_07 * buf2[local_y][local_x + 2];
            val += G_08 * buf2[local_y][local_x + 3];
            val += G_09 * buf2[local_y][local_x + 4];
            val += G_10 * buf2[local_y][local_x + 5];
        }
        buf3[local_y][local_x - 5] = val;
    }
}
inline void do_separable_conv_x_buf1(
    int groupThreadID_x,
    int groupThreadID_y,
    bool sq = false
)
{
    int local_y = groupThreadID_y;
    int local_x = groupThreadID_x + 5;
    float val = 0.0f;

    if (sq) {
        val += G_00 * do_sq(buf1[local_y][local_x - 5]);
        val += G_01 * do_sq(buf1[local_y][local_x - 4]);
        val += G_02 * do_sq(buf1[local_y][local_x - 3]);
        val += G_03 * do_sq(buf1[local_y][local_x - 2]);
        val += G_04 * do_sq(buf1[local_y][local_x - 1]);
        val += G_05 * do_sq(buf1[local_y][local_x]);
        val += G_06 * do_sq(buf1[local_y][local_x + 1]);
        val += G_07 * do_sq(buf1[local_y][local_x + 2]);
        val += G_08 * do_sq(buf1[local_y][local_x + 3]);
        val += G_09 * do_sq(buf1[local_y][local_x + 4]);
        val += G_10 * do_sq(buf1[local_y][local_x + 5]);
    }
    else {
        val += G_00 * buf1[local_y][local_x - 5];
        val += G_01 * buf1[local_y][local_x - 4];
        val += G_02 * buf1[local_y][local_x - 3];
        val += G_03 * buf1[local_y][local_x - 2];
        val += G_04 * buf1[local_y][local_x - 1];
        val += G_05 * buf1[local_y][local_x];
        val += G_06 * buf1[local_y][local_x + 1];
        val += G_07 * buf1[local_y][local_x + 2];
        val += G_08 * buf1[local_y][local_x + 3];
        val += G_09 * buf1[local_y][local_x + 4];
        val += G_10 * buf1[local_y][local_x + 5];
    }
    buf3[local_y][local_x - 5] = val;

    val = 0.0;
    local_y = groupThreadID_y + BY;
    if (local_y < SY) {
        if (sq) {
            val += G_00 * do_sq(buf1[local_y][local_x - 5]);
            val += G_01 * do_sq(buf1[local_y][local_x - 4]);
            val += G_02 * do_sq(buf1[local_y][local_x - 3]);
            val += G_03 * do_sq(buf1[local_y][local_x - 2]);
            val += G_04 * do_sq(buf1[local_y][local_x - 1]);
            val += G_05 * do_sq(buf1[local_y][local_x    ]);
            val += G_06 * do_sq(buf1[local_y][local_x + 1]);
            val += G_07 * do_sq(buf1[local_y][local_x + 2]);
            val += G_08 * do_sq(buf1[local_y][local_x + 3]);
            val += G_09 * do_sq(buf1[local_y][local_x + 4]);
            val += G_10 * do_sq(buf1[local_y][local_x + 5]);
        }
        else {
            val += G_00 * buf1[local_y][local_x - 5];
            val += G_01 * buf1[local_y][local_x - 4];
            val += G_02 * buf1[local_y][local_x - 3];
            val += G_03 * buf1[local_y][local_x - 2];
            val += G_04 * buf1[local_y][local_x - 1];
            val += G_05 * buf1[local_y][local_x    ];
            val += G_06 * buf1[local_y][local_x + 1];
            val += G_07 * buf1[local_y][local_x + 2];
            val += G_08 * buf1[local_y][local_x + 3];
            val += G_09 * buf1[local_y][local_x + 4];
            val += G_10 * buf1[local_y][local_x + 5];
        }
        buf3[local_y][local_x - 5] = val;
    }
}


inline float do_separable_conv_y_buf3(
    int groupThreadID_x,
    int groupThreadID_y,
)
{
    int local_y = groupThreadID_y + 5;
    int local_x = groupThreadID_x;
    float val = 0.0f;

    val += G_00 * buf3[local_y - 5][local_x];
    val += G_01 * buf3[local_y - 4][local_x];
    val += G_02 * buf3[local_y - 3][local_x];
    val += G_03 * buf3[local_y - 2][local_x];
    val += G_04 * buf3[local_y - 1][local_x];
    val += G_05 * buf3[local_y    ][local_x];
    val += G_06 * buf3[local_y + 1][local_x];
    val += G_07 * buf3[local_y + 2][local_x];
    val += G_08 * buf3[local_y + 3][local_x];
    val += G_09 * buf3[local_y + 4][local_x];
    val += G_10 * buf3[local_y + 5][local_x];

    return val;
}


[CudaKernel]
void FastSSIMForwardKernel(
    // input-buffers
    TensorView<float> img1, // [C,H,W]
    TensorView<float> img2, // [C,H,W]
    // input-scalars
    int H, int W,
    float C1, float C2,
    // output-buffers
    TensorView<float> dm_dmu1,       // [C,H,W]
    TensorView<float> dm_dsigma1_sq, // [C,H,W]
    TensorView<float> dm_dsigma12,   // [C,H,W]
    TensorView<float> ssim_map       // [C,H,W]
);

[[vk::push_constant]]
cbuffer PushConstants
{
    float xyz_size;
    float width;
    float height;
    float position_lr;
    float D;
};

[[vk::binding(0)]] StructuredBuffer<float> img1;
[[vk::binding(1)]] StructuredBuffer<float> img2;
[[vk::binding(2)]] RWStructuredBuffer<float> dm_dmu1;
[[vk::binding(3)]] RWStructuredBuffer<float> dm_dsigma1_sq;
[[vk::binding(4)]] RWStructuredBuffer<float> dm_dsigma12;
[[vk::binding(5)]] RWStructuredBuffer<float> ssim_map;
[[vk::binding(6)]] RWStructuredBuffer<float> dL_dmap;

[shader("compute")]
[numthreads(BX, BY, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint3 groupID = dispatchThreadID / uint3(BX, BY, 1);
    uint3 groupThreadID = dispatchThreadID % uint3(BX, BY, 1);
    int pix_y = groupID.y * BY + groupThreadID.y;
    int pix_x = groupID.x * BX + groupThreadID.x;
    // const int pix_id = pix_y * width + pix_x;
    // const int num_pix = height * width;
    int item_idx = groupThreadID.x + groupThreadID.y * BX;
    int H = int(height);
    int W = int(width);
    const float C1 = 0.0001f;
    const float C2 = 0.0009f;

    dL_dmap[t2i(0, pix_y, pix_x, H, W)] = -1.0 * lambda_dssim / (3 * H * W);
    dL_dmap[t2i(1, pix_y, pix_x, H, W)] = -1.0 * lambda_dssim / (3 * H * W);
    dL_dmap[t2i(2, pix_y, pix_x, H, W)] = -1.0 * lambda_dssim / (3 * H * W);

    for (int c = 0; c < 3; c++) {

        // load into shared
        // load_into_shared(buf1, img1, H, W, c, item_idx, groupID.x, groupID.y, dispatchThreadID.x, dispatchThreadID.y);
        load_into_shared_buf1(img1, H, W, c, item_idx, groupID.x, groupID.y, dispatchThreadID.x, dispatchThreadID.y);
        AllMemoryBarrierWithGroupSync();
        
        // calculate mu1
        // flush_conv_scratch(buf3, item_idx);
        flush_conv_scratch_buf3(item_idx);
        AllMemoryBarrierWithGroupSync();

        do_separable_conv_x_buf1(groupThreadID.x, groupThreadID.y, false);
        AllMemoryBarrierWithGroupSync();

        float mu1 = do_separable_conv_y_buf3(groupThreadID.x, groupThreadID.y);
        AllMemoryBarrierWithGroupSync();

        // calculate sigma1_sq
        // flush_conv_scratch(buf3, item_idx);
        flush_conv_scratch_buf3(item_idx);
        AllMemoryBarrierWithGroupSync();

        do_separable_conv_x_buf1(groupThreadID.x, groupThreadID.y, true);
        AllMemoryBarrierWithGroupSync();

        float sigma1_sq = do_separable_conv_y_buf3(groupThreadID.x, groupThreadID.y) - mu1 * mu1;
        AllMemoryBarrierWithGroupSync();

        // calculate mu2
        // load_into_shared(buf2, img2, H, W, c, item_idx, groupID.x, groupID.y, dispatchThreadID.x, dispatchThreadID.y);
        load_into_shared_buf2(img2, H, W, c, item_idx, groupID.x, groupID.y, dispatchThreadID.x, dispatchThreadID.y);
        AllMemoryBarrierWithGroupSync();

        // flush_conv_scratch(buf3, item_idx);
        flush_conv_scratch_buf3(item_idx);
        AllMemoryBarrierWithGroupSync();

        do_separable_conv_x_buf2(groupThreadID.x, groupThreadID.y, false);
        AllMemoryBarrierWithGroupSync();

        float mu2 = do_separable_conv_y_buf3(groupThreadID.x, groupThreadID.y);
        AllMemoryBarrierWithGroupSync();

        // calculate sigma2_sq
        // flush_conv_scratch(buf3, item_idx);
        flush_conv_scratch_buf3(item_idx);
        AllMemoryBarrierWithGroupSync();

        do_separable_conv_x_buf2(groupThreadID.x, groupThreadID.y, true);
        AllMemoryBarrierWithGroupSync();

        float sigma2_sq = do_separable_conv_y_buf3(groupThreadID.x, groupThreadID.y) - mu2 * mu2;
        AllMemoryBarrierWithGroupSync();

        // calculate sigma12
        multiply_shared_mem_buf1_buf2(item_idx);
        AllMemoryBarrierWithGroupSync();

        // flush_conv_scratch(buf3, item_idx);
        flush_conv_scratch_buf3(item_idx);
        AllMemoryBarrierWithGroupSync();

        do_separable_conv_x_buf1(groupThreadID.x, groupThreadID.y, false);
        AllMemoryBarrierWithGroupSync();

        float sigma12 = do_separable_conv_y_buf3(groupThreadID.x, groupThreadID.y) - mu1 * mu2;
        AllMemoryBarrierWithGroupSync();

        float mu1_sq = mu1 * mu1;
        float mu2_sq = mu2 * mu2;
        float mu1_mu2 = mu1 * mu2;
        float C = (2.0f * mu1_mu2 + C1);
        float D = (2.0f * sigma12 + C2);
        float A = (mu1_sq + mu2_sq + C1);
        float B = (sigma1_sq + sigma2_sq + C2);
        float m = (C * D) / (A * B);

        if (pix_x < width && pix_y < height) {
            // uint global_id = t2i(c, pix_y, pix_x, height, width);

            ssim_map[t2i(c, pix_y, pix_x, H, W)] = m;
            dm_dmu1[t2i(c, pix_y, pix_x, H, W)] = (
                (mu2 * 2.0f * D) / (A * B)
                - (mu2 * 2.0f * C) / (A * B)
                - (mu1 * 2.0f * C * D) / (A * A * B)
                + (mu1 * 2.0f * C * D) / (A * B * B)
            );
            dm_dsigma1_sq[t2i(c, pix_y, pix_x, H, W)] = ((-C * D) / (A * B * B));
            dm_dsigma12[t2i(c, pix_y, pix_x, H, W)] = ((2 * C) / (A * B));
        }

    }
}