#ifndef KERNELS
#define KERNELS
#define lgWARP 5
#define WARP (1 << lgWARP)

#include "cub.cuh"
#include <numeric>
// Give cred to the cheatsheet
// Gets global threadId of 2D grid of 2D blocks
__device__ int getGlobalIdx()
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

template <class T>
class Add
{
public:
    typedef T InpElTp;
    typedef T RedElTp;
    static const bool commutative = true;
    static __device__ __host__ inline T identInp() { return (T)0; }
    static __device__ __host__ inline T mapFun(const T &el) { return el; }
    static __device__ __host__ inline T identity() { return (T)0; }
    static __device__ __host__ inline T apply(const T t1, const T t2) { return t1 + t2; }

    static __device__ __host__ inline bool equals(const T t1, const T t2) { return (t1 == t2); }
    static __device__ __host__ inline T remVolatile(volatile T &t)
    {
        T res = t;
        return res;
    }
};

template <class OP>
__device__ inline typename OP::RedElTp
scanIncWarp(volatile typename OP::RedElTp *ptr, const unsigned int idx)
{
    const unsigned int lane = idx & (WARP - 1);
    // if(lane==0) {
    //     #pragma unroll
    //     for(int i=1; i<WARP; i++) {
    //         ptr[idx+i] = OP::apply(ptr[idx+i-1], ptr[idx+i]);
    //     }
    // }
    for (int i = 0; i < lgWARP; i++)
    {
        int h = 1 << i;

        if (lane >= h)
        {
            ptr[idx] = OP::apply(ptr[idx - h], ptr[idx]);
        }
    }
    return OP::remVolatile(ptr[idx]);
}

template <class OP>
__device__ inline typename OP::RedElTp
scanIncBlock(volatile typename OP::RedElTp *ptr, const unsigned int idx, int N)
{
    const unsigned int lane = idx & (WARP - 1);
    const unsigned int warpid = idx >> lgWARP;

    // 1. perform scan at warp level
    typename OP::RedElTp res = scanIncWarp<OP>(ptr, idx);
    __syncthreads();

    // 2. place the end-of-warp results in
    //   the first warp. This works because
    //   warp size = 32, and
    //   max block size = 32^2 = 1024

    // only the last thread runs the iff -> thus only 1023 will have a warpid of 31
    typename OP::RedElTp temp;
    if (lane == (WARP - 1))
    {
        temp = OP::remVolatile(ptr[idx]);
    }
    __syncthreads();
    if (lane == (WARP - 1))
    {
        ptr[warpid] = temp;
    }
    __syncthreads();

    // 3. scan again the first warp
    if (warpid == 0)
        scanIncWarp<OP>(ptr, idx);
    __syncthreads();

    // 4. accumulate results from previous step;
    if (warpid > 0)
    {
        res = OP::apply(ptr[warpid - 1], res);
    }
    __syncthreads();

    ptr[idx] = res;

    __syncthreads();
    return res;
}

template <class OP, int block_size>
__device__ void partition2(uint32_t *loc_data, uint32_t iteration, uint32_t bstart, uint32_t iter, uint32_t N)
{


    __shared__ uint32_t ps[1024];
    __shared__ uint32_t negPs[1024];


    const uint32_t loc_threadidx = (threadIdx.y * blockDim.x) + threadIdx.x;
    const uint32_t blockidx = blockIdx.x + blockIdx.y * gridDim.x;
    const uint32_t glb_threadidx = getGlobalIdx();

    const uint32_t mask = (1 << (bstart + iteration));
    uint32_t dat4;
    uint32_t p;
    uint32_t negP;
    uint32_t idx = loc_threadidx;

    __syncthreads();


    ps[idx] = 0;
    negPs[idx] = 0;

    if (glb_threadidx < N)
    {
    dat4 = loc_data[idx];


    p = (dat4 & mask) >> (iteration+iter*4);
    negP = 1 - p;

    ps[idx] = p;
    negPs[idx] = negP;
}




    __syncthreads();

    scanIncBlock<OP>(ps, idx, N);
    scanIncBlock<OP>(negPs, idx, N);




    __syncthreads();


    if (glb_threadidx < N)
    {
        int len_false = negPs[block_size - 1];

        int iT, iF;
        iT = ps[idx] -1 + len_false;
        iF = negPs[idx];


        if (p)
        {
            iT = ps[idx] -1 + len_false;
            loc_data[iT] = dat4;
        }
        else
        {
            iF = negPs[idx];
            loc_data[iF - 1] = dat4;
        }
    }




    __syncthreads();
    
}

// Exclusive scan with plus operator
template <class T>
__device__ void plus_scan(T *scanned_arr, T *arr, int n)
{
    scanned_arr[0] = 0;
    for (size_t i = 1; i < n; i++)
    {
        scanned_arr[i] = scanned_arr[i - 1] + arr[i - 1];
    }
}

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
    
    if (glb_threadidx < N){
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
    if (glb_threadidx < N){
        binidx = (data & mask) >> bstart;
        atomicAdd(&local_histogram[binidx], 1);
    }

    // __syncthreads();
    // if (loc_threadidx == 0 && blockidx == 0 && iter == 0){
    //     for (int i = 0; i < 16; i++)
    //     {
    //         printf("%d\n", local_histogram[i]);
    //     }
    //     printf("\n");
    // }

    __syncthreads();


    // SORT LOCAL TILE
    for (int i = 0; i < 4; i++)
    {
        partition2<Add<uint32_t>, block_size>(loc_data, i, bstart, iter, N);
    }

     __syncthreads();
    // if (loc_threadidx == 0 && blockidx == 0 && iter == 0)
    // {
    //     for (int i = 0; i < 1024; i++)
    //     {
    //         printf("%d\n", loc_data[i]);
    //     }
    // }
    // __syncthreads();

    // WRITE LOCAL HISTOGRAMS TO GLOBAL

    uint32_t p = gridDim.x * gridDim.y;
    if (glb_threadidx < N && loc_threadidx == 0){
        for (size_t i = 0; i < 16; i++)
        {
            glb_histogram[p * i + blockidx] = local_histogram[i];
        }
    }
    __syncthreads();

    // // WRITE SORTED TILE TO GLOBAL
    if (glb_threadidx < N){
        data_keys_out[glb_threadidx] = loc_data[loc_threadidx];
    }
    // __syncthreads();
    // if (loc_threadidx == 0 && blockidx == 0 && iter == 0){
    //     for (int i = 0; i < 64; i++)
    //     {
    //         printf("%d\n", glb_histogram[i]);
    //     }
    // }
}

template <int block_size> //<class ElTp> <- we will need this to generalize
__global__ void kern4(uint32_t *glb_histogram_in, uint32_t *glb_data_in, uint32_t *glb_data_out, int N, int iter, uint32_t *hist)
{

    __shared__ uint32_t loc_data[block_size];
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

    if (glb_threadidx < N){
        data = glb_data_in[glb_threadidx];
        loc_data[loc_threadidx] = data;
    }
        

    __syncthreads();

    //local hist
    if (glb_threadidx < N){
        for (int i = 0; i < 16; i++)
        {
            local_histogram[i] = hist[p * i + blockidx];
        }
    }
    
    __syncthreads();

    // scanned local hist
    plus_scan(scan_local_histogram, local_histogram, 16);

    // __syncthreads();
    // if (loc_threadidx == 0 && blockidx == 1 && iter == 0){
    //     for (int i = 0; i < block_size; i++)
    //     {
    //         printf("%d\n", loc_data[i]);
    //     }
    // }
    __syncthreads();


    if (glb_threadidx < N){
        // binOf function
        int bstart = iter * 4;
        uint32_t mask;
        mask = (15 << bstart);

        elmBin = (data & mask) >> bstart;    
        glbScanElm = glb_histogram_in[p * elmBin + blockidx]; //V
        //if (loc_threadidx == 5 && blockidx == 1 && iter == 0) printf("%d: %d: %d: %d\n", data, mask, elmBin, glbScanElm);

        //printf("%d\n",glbScanElm);

        loc_scan_offset = scan_local_histogram[elmBin];

        //if(blockidx == 1) printf("%d: %d: %d: %d: %d\n",glbScanElm + (loc_threadidx - loc_scan_offset), glbScanElm, loc_threadidx, loc_scan_offset, data);

        //if (glbScanElm + (loc_threadidx - loc_scan_offset) == 195) printf("%d: %d: %d: %d: %d: %d\n", data, loc_threadidx, loc_scan_offset, glbScanElm, blockidx, glbScanElm + loc_threadidx - loc_scan_offset);
        glb_data_out[glbScanElm + (loc_threadidx - loc_scan_offset)] = data;
    }

    // __syncthreads();
    // if (loc_threadidx == 0 && blockidx == 0 && iter == 0){
    //     for (int i = 0; i < 16; i++)
    //     {
    //         printf("%d: %d: %d\n", i, local_histogram[i]);
    //     }
    // }
    //     __syncthreads();
    // if (loc_threadidx == 0 && blockidx == 1 && iter == 0){
    //     for (int i = 0; i < 16; i++)
    //     {
    //         printf("%d: %d\n", i, local_histogram[i]);
    //     }
    // }
    // __syncthreads();
    // if (loc_threadidx == 0 && blockidx == 1 && iter == 0){
    //     for (int i = 0; i < 4*16; i++)
    //     {
    //         printf("%d: %d\n", i, glb_histogram_in[i]);
    //     }
    // }
}

#endif
