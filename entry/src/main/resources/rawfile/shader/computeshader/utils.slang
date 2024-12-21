#define EPS 1e-10
#define BLOCK_X 16
#define BLOCK_Y 16
#define PIXELS_PER_TILE (BLOCK_X * BLOCK_Y)

uint t2i(uint i, uint j, uint width) {
    return i * width + j;
}

uint t2i(uint c, uint i, uint j, uint height, uint width) {
    return c * height * width + i * width + j;
}
