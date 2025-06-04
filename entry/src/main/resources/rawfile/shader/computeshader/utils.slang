#define EPS 1e-10
#define BLOCK_X 16
#define BLOCK_Y 16
#define PIXELS_PER_TILE (BLOCK_X * BLOCK_Y)

static const uint toAddNum = 500;
static const float percent_dense = 0.01;
// static const float scene_extent = 1.0;
// static const float scene_extent = 5.202147674560547;
// static const float scene_extent = 5.202147674560547 / 5;
static const uint DD = 16; // the upper bound of D
static const float min_opacity = 0.005;
static const int max_screen_size = -1;
static const float lambda_dssim = 0.2;

uint t2i(uint i, uint j, uint width) {
    return i * width + j;
}

uint t2i(uint c, uint i, uint j, uint height, uint width) {
    return c * height * width + i * width + j;
}
void replaceGS(
    RWStructuredBuffer<float> xyz,
    RWStructuredBuffer<float> shs, int D,
    RWStructuredBuffer<float> opacity,
    RWStructuredBuffer<float> scale,
    RWStructuredBuffer<float> rotate,
    int32_t src_idx,
    int32_t dst_idx
)
{
    xyz[t2i(dst_idx, 0, 3)] = xyz[t2i(src_idx, 0, 3)];
    xyz[t2i(dst_idx, 1, 3)] = xyz[t2i(src_idx, 1, 3)];
    xyz[t2i(dst_idx, 2, 3)] = xyz[t2i(src_idx, 2, 3)];

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < D; j++) {
            shs[t2i(dst_idx, i, j, 3, DD)] = shs[t2i(src_idx, i, j, 3, DD)];
        }
    }

    opacity[dst_idx] = opacity[src_idx];

    scale[t2i(dst_idx, 0, 3)] = scale[t2i(src_idx, 0, 3)];
    scale[t2i(dst_idx, 1, 3)] = scale[t2i(src_idx, 1, 3)];
    scale[t2i(dst_idx, 2, 3)] = scale[t2i(src_idx, 2, 3)];

    rotate[t2i(dst_idx, 0, 4)] = rotate[t2i(src_idx, 0, 4)];
    rotate[t2i(dst_idx, 1, 4)] = rotate[t2i(src_idx, 1, 4)];
    rotate[t2i(dst_idx, 2, 4)] = rotate[t2i(src_idx, 2, 4)];
    rotate[t2i(dst_idx, 3, 4)] = rotate[t2i(src_idx, 3, 4)];
}

void replaceGS(
    RWStructuredBuffer<float> xyz,
    RWStructuredBuffer<float> shs, int D,
    RWStructuredBuffer<float> opacity,
    RWStructuredBuffer<float> scale,
    RWStructuredBuffer<float> rotate,
    RWStructuredBuffer<float> xyz_exp_avg,
    RWStructuredBuffer<float> scale_exp_avg,
    RWStructuredBuffer<float> shs_exp_avg,
    RWStructuredBuffer<float4> rotate_exp_avg,
    RWStructuredBuffer<float> opacity_exp_avg,
    RWStructuredBuffer<float> xyz_avg,
    RWStructuredBuffer<float> scale_avg,
    RWStructuredBuffer<float> shs_avg,
    RWStructuredBuffer<float4> rotate_avg,
    RWStructuredBuffer<float> opacity_avg,
    int32_t src_idx,
    int32_t dst_idx
)
{
    xyz[t2i(dst_idx, 0, 3)] = xyz[t2i(src_idx, 0, 3)];
    xyz[t2i(dst_idx, 1, 3)] = xyz[t2i(src_idx, 1, 3)];
    xyz[t2i(dst_idx, 2, 3)] = xyz[t2i(src_idx, 2, 3)];

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < D; j++) {
            shs[t2i(dst_idx, i, j, 3, DD)] = shs[t2i(src_idx, i, j, 3, DD)];
        }
    }

    opacity[dst_idx] = opacity[src_idx];

    scale[t2i(dst_idx, 0, 3)] = scale[t2i(src_idx, 0, 3)];
    scale[t2i(dst_idx, 1, 3)] = scale[t2i(src_idx, 1, 3)];
    scale[t2i(dst_idx, 2, 3)] = scale[t2i(src_idx, 2, 3)];

    rotate[t2i(dst_idx, 0, 4)] = rotate[t2i(src_idx, 0, 4)];
    rotate[t2i(dst_idx, 1, 4)] = rotate[t2i(src_idx, 1, 4)];
    rotate[t2i(dst_idx, 2, 4)] = rotate[t2i(src_idx, 2, 4)];
    rotate[t2i(dst_idx, 3, 4)] = rotate[t2i(src_idx, 3, 4)];

    // Replace Adam's first moment (exp_avg)
    xyz_exp_avg[t2i(dst_idx, 0, 3)] = xyz_exp_avg[t2i(src_idx, 0, 3)];
    xyz_exp_avg[t2i(dst_idx, 1, 3)] = xyz_exp_avg[t2i(src_idx, 1, 3)];
    xyz_exp_avg[t2i(dst_idx, 2, 3)] = xyz_exp_avg[t2i(src_idx, 2, 3)];

    scale_exp_avg[t2i(dst_idx, 0, 3)] = scale_exp_avg[t2i(src_idx, 0, 3)];
    scale_exp_avg[t2i(dst_idx, 1, 3)] = scale_exp_avg[t2i(src_idx, 1, 3)];
    scale_exp_avg[t2i(dst_idx, 2, 3)] = scale_exp_avg[t2i(src_idx, 2, 3)];

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < D; j++) {
            shs_exp_avg[t2i(dst_idx, i, j, 3, DD)] = shs_exp_avg[t2i(src_idx, i, j, 3, DD)];
        }
    }

    rotate_exp_avg[t2i(dst_idx, 0, 4)] = rotate_exp_avg[t2i(src_idx, 0, 4)];
    rotate_exp_avg[t2i(dst_idx, 1, 4)] = rotate_exp_avg[t2i(src_idx, 1, 4)];
    rotate_exp_avg[t2i(dst_idx, 2, 4)] = rotate_exp_avg[t2i(src_idx, 2, 4)];
    rotate_exp_avg[t2i(dst_idx, 3, 4)] = rotate_exp_avg[t2i(src_idx, 3, 4)];

    opacity_exp_avg[dst_idx] = opacity_exp_avg[src_idx];

    // Replace Adam's second moment (avg)
    xyz_avg[t2i(dst_idx, 0, 3)] = xyz_avg[t2i(src_idx, 0, 3)];
    xyz_avg[t2i(dst_idx, 1, 3)] = xyz_avg[t2i(src_idx, 1, 3)];
    xyz_avg[t2i(dst_idx, 2, 3)] = xyz_avg[t2i(src_idx, 2, 3)];

    scale_avg[t2i(dst_idx, 0, 3)] = scale_avg[t2i(src_idx, 0, 3)];
    scale_avg[t2i(dst_idx, 1, 3)] = scale_avg[t2i(src_idx, 1, 3)];
    scale_avg[t2i(dst_idx, 2, 3)] = scale_avg[t2i(src_idx, 2, 3)];

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < D; j++) {
            shs_avg[t2i(dst_idx, i, j, 3, DD)] = shs_avg[t2i(src_idx, i, j, 3, DD)];
        }
    }

    rotate_avg[t2i(dst_idx, 0, 4)] = rotate_avg[t2i(src_idx, 0, 4)];
    rotate_avg[t2i(dst_idx, 1, 4)] = rotate_avg[t2i(src_idx, 1, 4)];
    rotate_avg[t2i(dst_idx, 2, 4)] = rotate_avg[t2i(src_idx, 2, 4)];
    rotate_avg[t2i(dst_idx, 3, 4)] = rotate_avg[t2i(src_idx, 3, 4)];

    opacity_avg[dst_idx] = opacity_avg[src_idx];
}

void resetGS(
    RWStructuredBuffer<float> xyz,
    RWStructuredBuffer<float> shs, int D,
    RWStructuredBuffer<float> opacity,
    RWStructuredBuffer<float> scale,
    RWStructuredBuffer<float> rotate,
    int32_t idx,
)
{
    xyz[t2i(idx, 0, 3)] = 0;
    xyz[t2i(idx, 1, 3)] = 0;
    xyz[t2i(idx, 2, 3)] = 0;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < D; j++) {
            shs[t2i(idx, i, j, 3, DD)] = 0;
        }
    }

    opacity[idx] = 0;

    scale[t2i(idx, 0, 3)] = 0;
    scale[t2i(idx, 1, 3)] = 0;
    scale[t2i(idx, 2, 3)] = 0;

    rotate[t2i(idx, 0, 4)] = 0;
    rotate[t2i(idx, 1, 4)] = 0;
    rotate[t2i(idx, 2, 4)] = 0;
    rotate[t2i(idx, 3, 4)] = 0;
}
