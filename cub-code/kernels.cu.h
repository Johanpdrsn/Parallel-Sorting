#ifndef KERNELS
#define KERNELS

#include "cub.cuh"
#include <numeric>

template <int block_size> //<class ElTp> <- we will need this to generalize
__global__ void kern1(uint32_t *data_keys_in, uint32_t *data_keys_out, uint32_t *glb_histogram, int N, int iter)
{
    __shared__ uint32_t loc_data[block_size];
    __shared__ uint32_t local_histogram[16];

    uint32_t data;

    //memset here?

    const int blockidx = blockIdx.x + blockIdx.y * gridDim.x;
    const int glb_threadidx = getGlobalIdx();
    const int loc_threadidx = (threadIdx.y * blockDim.x) + threadIdx.x;

    if (loc_threadidx < 16)
        local_histogram[loc_threadidx] = 0;

    // This is not coalesced on memory (we should stride instead of taking 4 seq)

    // Read data from global memory.
    // Loop over 4 data entries.
    if (glb_threadidx < N)
    {
        data = data_keys_in[glb_threadidx];
        loc_data[loc_threadidx] = data;
    }

    // Sync needed after loading all data in.
    __syncthreads();

    int bstart = iter * 4;
    int binidx;

    uint32_t mask;
    mask = (15 << bstart);
    // Loop over 4 data entries
    if (glb_threadidx < N)
    {
        binidx = (data & mask) >> bstart;
        atomicAdd(&local_histogram[binidx], 1);
    }

    __syncthreads();

    // SORT LOCAL TILE
    for (int i = 0; i < 4; i++)
    {
        partition2<Add<uint32_t>, block_size>(loc_data, i, bstart, iter, N);
    }

    __syncthreads();

    uint32_t p = gridDim.x * gridDim.y;
    if (glb_threadidx < N && loc_threadidx == 0)
    {
        for (size_t i = 0; i < 16; i++)
        {
            glb_histogram[p * i + blockidx] = local_histogram[i];
        }
    }
    __syncthreads();

    // // WRITE SORTED TILE TO GLOBAL
    if (glb_threadidx < N)
    {
        data_keys_out[glb_threadidx] = loc_data[loc_threadidx];
    }
}

template <int block_size> //<class ElTp> <- we will need this to generalize
__global__ void kern1_tiled(uint32_t *data_keys_in, uint32_t *data_keys_out, uint32_t *glb_histogram, int N, int iter)
{
    __shared__ uint32_t loc_data[block_size];
    __shared__ uint32_t local_histogram[16];

    uint32_t data[4];

    //memset here?

    const int blockidx = blockIdx.x + blockIdx.y * gridDim.x;
    const int glb_threadidx = getGlobalIdx();
    const int loc_threadidx = (threadIdx.y * blockDim.x) + threadIdx.x;

    const int glb_memoffset = 4 * glb_threadidx;
    const int loc_memoffset = 4 * loc_threadidx;

    if (loc_threadidx < 16)
        local_histogram[loc_threadidx] = 0;

    // This is not coalesced on memory (we should stride instead of taking 4 seq)

    // Read data from global memory.
    // Loop over 4 data entries.
    for (int i = 0; i < 4; i++)
    {
        if (glb_memoffset + i < N)
        {
            data[i] = data_keys_in[glb_memoffset + i];
            loc_data[loc_memoffset + i] = data[i];
        }
    }
    // Sync needed after loading all data in.
    __syncthreads();

    int bstart = iter * 4;
    int binidx;

    uint32_t mask;
    mask = (15 << bstart);
    // Loop over 4 data entries
    for (int i = 0; i < 4; i++)
    {
        if (glb_memoffset + i < N)
        {
            binidx = (data[i] & mask) >> bstart;
            atomicAdd(&local_histogram[binidx], 1);
        }
    }
    __syncthreads();

    // // WRITE SORTED TILE TO GLOBAL
    for (int i = 0; i < 4; i++)
    {
        if (glb_memoffset + i < N)
        {
            data_keys_out[glb_memoffset + i] = loc_data[loc_memoffset + i];
        }
    }
    __syncthreads();

    // SORT LOCAL TILE
    for (int i = 0; i < 4; i++)
    {
        partition2_tiled<Add<uint32_t>, block_size>(loc_data, i, bstart, iter, N);
    }

    __syncthreads();

    uint32_t p = gridDim.x * gridDim.y;

    if (loc_threadidx == 0)
    {
        for (size_t i = 0; i < 16; i++)
        {
            glb_histogram[p * i + blockidx] = local_histogram[i];
        }
    }

    __syncthreads();
}

template <int block_size> //<class ElTp> <- we will need this to generalize
__global__ void kern4(uint32_t *glb_histogram_in, uint32_t *glb_data_in, uint32_t *glb_data_out, int N, int iter, uint32_t *hist)
{

    __shared__ uint32_t local_histogram[16];

    uint32_t scan_local_histogram[16];

    uint32_t data;

    const int blockidx = blockIdx.x + blockIdx.y * gridDim.x;
    const int glb_threadidx = getGlobalIdx();
    const int loc_threadidx = (threadIdx.y * blockDim.x) + threadIdx.x;

    if (loc_threadidx < 16)
        local_histogram[loc_threadidx] = 0;

    // This is not coalesced on memory (we should stride instead of taking 4 seq)
    const int p = gridDim.x * gridDim.y;

    int32_t elmBin;
    int32_t glbScanElm;
    int32_t loc_scan_offset;
    // Read data from global memory.
    // Loop over 4 data entries.

    if (glb_threadidx < N)
    {
        data = glb_data_in[glb_threadidx];
    }

    __syncthreads();

    //local hist
    if (glb_threadidx < N)
    {
        for (int i = 0; i < 16; i++)
        {
            local_histogram[i] = hist[p * i + blockidx];
        }
    }

    __syncthreads();

    // scanned local hist
    plus_scan(scan_local_histogram, local_histogram, 16);

    __syncthreads();

    if (glb_threadidx < N)
    {
        // binOf function
        int bstart = iter * 4;
        uint32_t mask;
        mask = (15 << bstart);

        elmBin = (data & mask) >> bstart;
        glbScanElm = glb_histogram_in[p * elmBin + blockidx]; //V

        loc_scan_offset = scan_local_histogram[elmBin];

        glb_data_out[glbScanElm + (loc_threadidx - loc_scan_offset)] = data;
    }
}

template <int block_size> //<class ElTp> <- we will need this to generalize
__global__ void kern4_tiled(uint32_t *glb_histogram_in, uint32_t *glb_data_in, uint32_t *glb_data_out, int N, int iter, uint32_t *hist)
{

    __shared__ uint32_t local_histogram[16];
    uint32_t scan_local_histogram[16];

    uint32_t data[4];

    const int blockidx = blockIdx.x + blockIdx.y * gridDim.x;
    const int glb_threadidx = getGlobalIdx();
    const int loc_threadidx = (threadIdx.y * blockDim.x) + threadIdx.x;

    if (loc_threadidx < 16)
        local_histogram[loc_threadidx] = 0;

    // This is not coalesced on memory (we should stride instead of taking 4 seq)
    const int p = gridDim.x * gridDim.y;

    int32_t elmBin;
    int32_t glbScanElm;
    int32_t loc_scan_offset;

    const int glb_memoffset = 4 * glb_threadidx;
    const int loc_memoffset = 4 * loc_threadidx;
    // Read data from global memory.
    // Loop over 4 data entries.
    for (int i = 0; i < 4; i++)
    {
        if (glb_memoffset + i < N)
        {
            data[i] = glb_data_in[glb_memoffset + i];
        }
    }
    __syncthreads();

    //local hist
    for (int i = 0; i < 4; i++)
    {
        if (glb_memoffset + i < N && loc_threadidx == 0)
        {
            for (int i = 0; i < 16; i++)
            {
                local_histogram[i] = hist[p * i + blockidx];
            }
        }
    }

    __syncthreads();

    // scanned local hist
    plus_scan(scan_local_histogram, local_histogram, 16);

    __syncthreads();

    for (int i = 0; i < 4; i++)
    {
        if (glb_memoffset + i < N)
        {
            // binOf function
            int bstart = iter * 4;
            uint32_t mask;
            mask = (15 << bstart);

            elmBin = (data[i] & mask) >> bstart;
            glbScanElm = glb_histogram_in[p * elmBin + blockidx]; //V

            loc_scan_offset = scan_local_histogram[elmBin];

            glb_data_out[glbScanElm + (loc_memoffset + i - loc_scan_offset)] = data[i];
        }
    }
}

#endif
