[CudaKernel]
void computeTileRangeKernel(
    TensorView<int64_t> sorted_keys_array,
    TensorView<int32_t> unsorted_idx_array,
    TensorView<int32_t> indices_array,
    // outputs
    TensorView<int32_t> tile_range_array,
    TensorView<int32_t> sorted_idx_array
);

// Buffer bindings
[[vk::binding(0)]] StructuredBuffer<uint32_t> sorted_keys_array;
[[vk::binding(1)]] StructuredBuffer<int32_t> unsorted_idx_array;
[[vk::binding(2)]] StructuredBuffer<int32_t> indices_array;
[[vk::binding(3)]] StructuredBuffer<int32_t> sort_keys_size;

[[vk::binding(4)]] RWStructuredBuffer<int32_t> tile_range_array;
[[vk::binding(5)]] RWStructuredBuffer<int32_t> sorted_idx_array;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint xyz_size;
    uint W;
    uint H;
    uint temp;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint32_t idx = dispatchThreadID.x;
    if (idx.x >= sort_keys_size[0])
        return;

    uint32_t currtile = uint32_t((sorted_keys_array[idx]) >> 19);
    bool valid_tile = (currtile != 10000000);

    if (valid_tile) 
        sorted_idx_array[idx] = indices_array[idx];
    

    if (idx == 0) 
        tile_range_array[currtile * 2] = 0;

    else
    {
        uint32_t prevtile = uint32_t((sorted_keys_array[idx - 1]) >> 19);
        if (currtile != prevtile) {
            tile_range_array[prevtile * 2 + 1] = idx;
            if (valid_tile) {
                tile_range_array[currtile * 2] = idx;
            }
        }
    }

    if (idx == sort_keys_size[0] - 1 && valid_tile) {
        tile_range_array[currtile * 2 + 1] = sort_keys_size[0];
    }

}
