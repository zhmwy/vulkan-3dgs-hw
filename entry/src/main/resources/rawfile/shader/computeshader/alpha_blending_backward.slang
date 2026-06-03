#include "utils.slang"

#define SHARE_CHUNK_SIZE 32

static const float SCALE = 1.0e8;
static const float SCALE_UVD = 1.0e13;
static const float SCALE_CONIC = 1.0e7;
static const float SCALE_OPACITY = 1.0e9;
static const float SCALE_FEATURE = 1.0e9;

// Buffer bindings
[[vk::binding(0)]] StructuredBuffer<float> uvd_array;
[[vk::binding(1)]] StructuredBuffer<float> conic_array;
[[vk::binding(2)]] StructuredBuffer<float> opacity_array;
[[vk::binding(3)]] StructuredBuffer<float> feature_array;
[[vk::binding(4)]] StructuredBuffer<int> sort_idx_array;
[[vk::binding(5)]] StructuredBuffer<int> tile_range_array;
[[vk::binding(6)]] StructuredBuffer<float> background_array;
[[vk::binding(8)]] StructuredBuffer<float> final_T_array;
[[vk::binding(9)]] StructuredBuffer<int32_t> ncontrib_array;

[[vk::binding(10)]] StructuredBuffer<float> dL_drfeat_array;
[[vk::binding(11)]] RWByteAddressBuffer dL_duvd_array;
[[vk::binding(12)]] RWByteAddressBuffer dL_dconic_array;
[[vk::binding(13)]] RWByteAddressBuffer dL_dopacity_array;
[[vk::binding(14)]] RWByteAddressBuffer dL_dfeature_array;
[[vk::binding(15)]] RWByteAddressBuffer dL_dndc_array;

[[vk::binding(16)]] RWStructuredBuffer<float> dL_dbackground_array;


groupshared int32_t shared_idx[SHARE_CHUNK_SIZE];
groupshared float3 shared_uvd[SHARE_CHUNK_SIZE];
groupshared float3 shared_invcov[SHARE_CHUNK_SIZE];
groupshared float shared_opacity[SHARE_CHUNK_SIZE];
groupshared float3 shared_feature[SHARE_CHUNK_SIZE];

// groupshared int32_t shared_dL_duv[SHARE_CHUNK_SIZE * 2];
// groupshared int32_t shared_dL_dconic[SHARE_CHUNK_SIZE * 3];
// groupshared int32_t shared_dL_dopacity[SHARE_CHUNK_SIZE];
// groupshared int32_t shared_dL_dfeature[SHARE_CHUNK_SIZE * 3];

[[vk::push_constant]]
cbuffer PushConstants
{
    uint xyz_size;
    uint W;
    uint H;
    uint training_step;
};

[shader("compute")]
[numthreads(BLOCK_X, BLOCK_Y, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    int32_t gridDim_x = (W + BLOCK_X - 1) / BLOCK_X;
    int32_t tile_idx_x = dispatchThreadID.x / BLOCK_X;
    int32_t tile_idx_y = dispatchThreadID.y / BLOCK_Y;
    int32_t tile_idx = tile_idx_y * gridDim_x + tile_idx_x;

    // index of shared gaussian in a chunk
    int32_t local_x = dispatchThreadID.x % BLOCK_X;
    int32_t local_y = dispatchThreadID.y % BLOCK_Y;
    int32_t item_idx = local_x + local_y * BLOCK_X;

    // pixel col and row in the image
    uint2 p = {
        tile_idx_x * BLOCK_X + local_x,
        tile_idx_y * BLOCK_Y + local_y
    };

    bool withinBounds = p.x < W && p.y < H;

    int2 gaussian_range = {
        tile_range_array[2 * tile_idx],
        tile_range_array[2 * tile_idx + 1]
    };
    int32_t gaussian_num = gaussian_range.y - gaussian_range.x;
    int32_t num_chunk = (gaussian_num + SHARE_CHUNK_SIZE - 1) / SHARE_CHUNK_SIZE;

    float3 dL_duvd_reg;
    float3 dL_dconic_reg;
    float dL_dopacity_reg;

    float final_trans = withinBounds ? float(final_T_array[t2i(p.y, p.x, W)]) : 0.0;
    float trans = final_trans;
    int remain_gaussian_num = gaussian_num;
    int current_process_idx = gaussian_num;
    int last_involved = withinBounds ? ncontrib_array[t2i(p.y, p.x, W)] : 0;

    int skipCount = gaussian_num - last_involved;

    float3 accum_feat_term = float3(0.0, 0.0, 0.0);

    for (int i = 0; i < num_chunk; i++, remain_gaussian_num -= SHARE_CHUNK_SIZE) {

        AllMemoryBarrierWithGroupSync();
        int32_t num_loaded = i * SHARE_CHUNK_SIZE + item_idx;

        if (gaussian_range.x + num_loaded < gaussian_range.y && item_idx < SHARE_CHUNK_SIZE) {

            int32_t gidx = sort_idx_array[gaussian_range.y - num_loaded - 1];

            shared_idx[item_idx] = gidx;

            shared_uvd[item_idx] = float3(
                float(uvd_array[3 * gidx]),
                float(uvd_array[3 * gidx + 1]),
                float(uvd_array[3 * gidx + 2])
            );

            shared_invcov[item_idx] = float3(
                float(conic_array[3 * gidx]),
                float(conic_array[3 * gidx + 1]),
                float(conic_array[3 * gidx + 2])
            );

            shared_opacity[item_idx] = float(opacity_array[gidx]);

            shared_feature[item_idx] = float3(
                float(feature_array[3 * gidx]),
                float(feature_array[3 * gidx + 1]),
                float(feature_array[3 * gidx + 2])
            );

        }
        AllMemoryBarrierWithGroupSync();

        int num_to_process = min(SHARE_CHUNK_SIZE, remain_gaussian_num);

        int32_t j = skipCount;
        skipCount = max(0, skipCount - num_to_process);

        for (; j < num_to_process; j++){
            float2 vec = float2(
                shared_uvd[j].x - float(p.x),
                shared_uvd[j].y - float(p.y)
            );

            // squared Mahalanobis distance
            float3 shared_invcov_reg = shared_invcov[j];
            float maha_distance = shared_invcov_reg.x * vec.x * vec.x
                                 + shared_invcov_reg.y * vec.x * vec.y * 2.0
                                 + shared_invcov_reg.z * vec.y * vec.y;

            float power = -0.5 * maha_distance;
            if (power > 0.0) {
                continue;
            }

            float G = exp(power);
            float current_opacity = shared_opacity[j];
            float alpha = min(0.99, current_opacity * G);
            if (alpha < 1.0 / 255.0) {
                continue;
            }

            float tmp = 1.0 / (1.0 - alpha);
            trans = trans * tmp;

            float3 dL_dalpha = float3(0.0, 0.0, 0.0);
            float3 dL_dG = float3(0.0, 0.0, 0.0);

            float2 dG_dvec = float2(
                -G * vec.x * shared_invcov_reg.x - G * vec.y * shared_invcov_reg.y,
                -G * vec.y * shared_invcov_reg.z - G * vec.x * shared_invcov_reg.y
            );

            dL_duvd_reg = float3(0.0);
            dL_dconic_reg = float3(0.0);
            dL_dopacity_reg = 0.0;

            // MTR
            // from feature_array
            float3 current_feature = shared_feature[j];
            float3 dL_drfeat_reg = float3(
                float(dL_drfeat_array[t2i(uint(0), p.y, p.x, H, W)]),
                float(dL_drfeat_array[t2i(uint(1), p.y, p.x, H, W)]),
                float(dL_drfeat_array[t2i(uint(2), p.y, p.x, H, W)])
            );
            dL_dalpha = (current_feature * trans - accum_feat_term * tmp) * dL_drfeat_reg;
            accum_feat_term += alpha * current_feature * trans;

            dL_dG = current_opacity * dL_dalpha;

            dL_duvd_reg.x += (dL_dG.x + dL_dG.y + dL_dG.z) * dG_dvec.x;
            dL_duvd_reg.y += (dL_dG.x + dL_dG.y + dL_dG.z) * dG_dvec.y;

            dL_dconic_reg.x += -0.5 * G * vec.x * vec.x * (dL_dG.x + dL_dG.y + dL_dG.z);
            dL_dconic_reg.y += -G * vec.x * vec.y * (dL_dG.x + dL_dG.y + dL_dG.z);
            dL_dconic_reg.z += -0.5 * G * vec.y * vec.y * (dL_dG.x + dL_dG.y + dL_dG.z);
            dL_dopacity_reg += G * (dL_dalpha.x + dL_dalpha.y + dL_dalpha.z);

            dL_drfeat_reg = alpha * trans * dL_drfeat_reg;

            int32_t gidx = shared_idx[j];
            dL_dfeature_array.InterlockedAddF32((gidx * 3 + 0) * 4, dL_drfeat_reg.x);
            dL_dfeature_array.InterlockedAddF32((gidx * 3 + 1) * 4, dL_drfeat_reg.y);
            dL_dfeature_array.InterlockedAddF32((gidx * 3 + 2) * 4, dL_drfeat_reg.z);

            dL_dconic_array.InterlockedAddF32(gidx * 3 * 4 + 0, dL_dconic_reg.x);
            dL_dconic_array.InterlockedAddF32(gidx * 3 * 4 + 1, dL_dconic_reg.y);
            dL_dconic_array.InterlockedAddF32(gidx * 3 * 4 + 2, dL_dconic_reg.z);

            dL_dndc_array.InterlockedAddF32((gidx * 3 + 0) * 4, dL_duvd_reg.x);
            dL_dndc_array.InterlockedAddF32((gidx * 3 + 1) * 4, dL_duvd_reg.y);

            dL_dopacity_array.InterlockedAddF32(gidx * 4, dL_dopacity_reg);
        }

    }

}
