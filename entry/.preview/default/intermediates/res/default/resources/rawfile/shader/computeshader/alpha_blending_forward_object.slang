#include "utils.slang"

#define SHARE_CHUNK_SIZE 32

// groupshared int32_t shared_idx[SHARE_CHUNK_SIZE];
groupshared half3 shared_uvd[SHARE_CHUNK_SIZE];
groupshared half3 shared_invcov[SHARE_CHUNK_SIZE];
groupshared half shared_opacity[SHARE_CHUNK_SIZE];
groupshared half3 shared_feature[SHARE_CHUNK_SIZE];

// Buffer bindings
[[vk::binding(0)]] StructuredBuffer<float> uvd_array;
[[vk::binding(1)]] StructuredBuffer<float> conic_array;
[[vk::binding(2)]] StructuredBuffer<float> opacity_array;
[[vk::binding(3)]] StructuredBuffer<float> feature_array;
[[vk::binding(4)]] StructuredBuffer<int> sort_idx_array;
[[vk::binding(5)]] StructuredBuffer<int> tile_range_array;
[[vk::binding(6)]] StructuredBuffer<float> background_array;

[[vk::binding(7)]] RWStructuredBuffer<float> rfeat_array;
[[vk::binding(8)]] RWStructuredBuffer<float> final_T_array;
[[vk::binding(9)]] RWStructuredBuffer<int32_t> ncontrib_array;

[[vk::binding(17)]] RWStructuredBuffer<float> alpha_array;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint xyz_size;
    uint width;
    uint height;
    uint training_step;
};

[shader("compute")]
[numthreads(BLOCK_X, BLOCK_Y, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    int32_t gridDim_x = (width + BLOCK_X - 1) / BLOCK_X;
    // index of tile
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

    bool withinBounds = p.x < width && p.y < height;
    bool thread_active = withinBounds;

    // Process gaussians in a tile in Chunks.
    // Each thread load only one Gaussian into shared memory.
    int2 gaussian_range = {
        tile_range_array[2 * tile_idx],
        tile_range_array[2 * tile_idx + 1]
    };
    int32_t gaussian_num = gaussian_range.y - gaussian_range.x;
    int32_t num_chunk = (gaussian_num + SHARE_CHUNK_SIZE - 1) / SHARE_CHUNK_SIZE;

    half trans = 1.0hf;
    uint32_t last_involved = 0;
    uint32_t current_process_idx = 0;

    half3 Feat = { 0.0hf, 0.0hf, 0.0hf };
    half mask = 0.0hf;
    int32_t remain_gaussian_num = gaussian_num;

    // process in chunks if there are too many Gaussians in a single tile
    for (int32_t i = 0; i < num_chunk; i++, remain_gaussian_num -= SHARE_CHUNK_SIZE) {

        // Collectively fetch per-Gaussian data from global to shared
        AllMemoryBarrierWithGroupSync();
        int32_t num_loaded = i * SHARE_CHUNK_SIZE + item_idx;
        if (gaussian_range.x + num_loaded < gaussian_range.y && item_idx < SHARE_CHUNK_SIZE) {
            // item_idx: index of gaussian in the chunk
            // gidx: gaussian index in the global array

            int32_t gidx = sort_idx_array[gaussian_range.x + num_loaded];

            shared_uvd[item_idx] = half3(
                half(uvd_array[3 * gidx]),
                half(uvd_array[3 * gidx + 1]),
                half(uvd_array[3 * gidx + 2])
            );

            shared_invcov[item_idx] = half3(
                half(conic_array[3 * gidx]),
                half(conic_array[3 * gidx + 1]),
                half(conic_array[3 * gidx + 2])
            );

            shared_opacity[item_idx] = half(opacity_array[gidx]);

            shared_feature[item_idx] = half3(
                half(feature_array[3 * gidx]),
                half(feature_array[3 * gidx + 1]),
                half(feature_array[3 * gidx + 2])
            );
        }
        AllMemoryBarrierWithGroupSync();

        int32_t num_to_process = min(SHARE_CHUNK_SIZE, remain_gaussian_num);

        if (thread_active) {
            for (int32_t j = 0; j < num_to_process; j++) {

                current_process_idx++;
                half2 vec = {
                    shared_uvd[j].x - half(p.x),
                    shared_uvd[j].y - half(p.y)
                };

                // squared Mahalanobis distance
                half maha_distance = shared_invcov[j].x * vec.x * vec.x
                                    + shared_invcov[j].y * vec.x * vec.y * 2.0hf
                                    + shared_invcov[j].z * vec.y * vec.y;

                half power = -0.5hf * maha_distance;

                if (power > 0.0hf)
                    continue;

                // alpha pruning
                half alpha;
                if (training_step > 0)
                    alpha = min(0.99hf, shared_opacity[j] * exp(power));
                else
                    alpha = min(0.99hf, shared_opacity[j] * exp(power));

                if (alpha < 1.0 / 255.0f)
                    continue;

                half next_trans = trans * (1.0hf - alpha);
                if (next_trans < 0.0001hf) {
                    thread_active = false;
                    break;
                }

                Feat += shared_feature[j] * alpha * trans;
                mask += alpha * trans;
                trans = next_trans;
                last_involved = current_process_idx;
            }
        }
    }

    if (withinBounds) {

        final_T_array[p.y * width + p.x] = trans;
        ncontrib_array[p.y * width + p.x] = last_involved;
        rfeat_array[t2i(0, p.y, p.x, height, width)] = Feat.x ; //+ trans * background_array[t2i(0, p.y, p.x, height, width)];
        rfeat_array[t2i(1, p.y, p.x, height, width)] = Feat.y ; //+ trans * background_array[t2i(1, p.y, p.x, height, width)];
        rfeat_array[t2i(2, p.y, p.x, height, width)] = Feat.z ; //+ trans * background_array[t2i(2, p.y, p.x, height, width)];
        
        mask = clamp(mask, half(1e-6), half(1 - 1e-6));
        alpha_array[t2i(0, p.y, p.x, height, width)] = mask;

    }
}