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
[[vk::binding(0)]] StructuredBuffer<int32_t> sorted_keys_array;
[[vk::binding(1)]] StructuredBuffer<int32_t> unsorted_idx_array;
[[vk::binding(2)]] StructuredBuffer<int32_t> indices_array;
[[vk::binding(3)]] StructuredBuffer<int32_t> sort_keys_size;

[[vk::binding(4)]] RWStructuredBuffer<int32_t2> tile_range_array;
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

    sorted_idx_array[idx] = indices_array[idx];

    uint32_t currtile = uint32_t((sorted_keys_array[idx]) >> 19);
    // tile_range_array[idx].x = currtile;return;

    if (idx == 0)
        tile_range_array[currtile].x = 0;
    else
    {
        uint32_t prevtile = uint32_t((sorted_keys_array[idx - 1]) >> 19);
        if (currtile != prevtile)
        {
            tile_range_array[prevtile].y = idx;
            tile_range_array[currtile].x = idx;
        }
    }
    if (idx == sort_keys_size[0] - 1)
        tile_range_array[currtile].y = sort_keys_size[0];
}
