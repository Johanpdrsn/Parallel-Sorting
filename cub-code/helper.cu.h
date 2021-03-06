#ifndef HISTO_HELPER
#define HISTO_HELPER

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#define GPU_RUNS 100
#define lgWARP 5
#define WARP (1 << lgWARP)

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution = 1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff < 0);
}

#define cudaCheckError()                                                                     \
    {                                                                                        \
        cudaError_t e = cudaGetLastError();                                                  \
        if (e != cudaSuccess)                                                                \
        {                                                                                    \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(0);                                                                         \
        }                                                                                    \
    }

#define cudaSucceeded(ans)                     \
    {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cerr << "cudaAssert failed: "
                  << cudaGetErrorString(code)
                  << file << ":" << line
                  << std::endl;
        if (abort)
        {
            exit(code);
        }
    }
}

inline uint32_t ceilLog2(uint32_t H)
{
    if (H == 0)
    {
        printf("Log2(0) is illegal. Exiting!\n");
        exit(1);
    }
    uint32_t log2_val = 0, pow2_val = 1;
    while (pow2_val < H)
    {
        log2_val++;
        pow2_val *= 2;
    }
    return log2_val;
}

void writeRuntime(const char *fname, double elapsed)
{
    FILE *f = fopen(fname, "w");
    assert(f != NULL);
    fprintf(f, "%f", elapsed);
    fclose(f);
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
scanIncBlock(volatile typename OP::RedElTp *ptr, const unsigned int idx)
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

    const int blockidx = blockIdx.x + blockIdx.y * gridDim.x;
    const int loc_threadidx = (threadIdx.y * blockDim.x) + threadIdx.x;
    const int glb_threadidx = blockidx * (blockDim.x * blockDim.y) + loc_threadidx;

    const uint32_t mask = (1 << (bstart + iteration));
    uint32_t dat4;
    uint32_t p;
    uint32_t negP;
    uint32_t idx = loc_threadidx;

    __syncthreads();

    ps[idx] = 0;
    negPs[idx] = 0;

    // CREATE BIT ARRAYS
    if (glb_threadidx < N)
    {
        dat4 = loc_data[idx];

        p = (dat4 & mask) >> (iteration + iter * 4);
        negP = 1 - p;

        ps[idx] = p;
        negPs[idx] = negP;
    }

    __syncthreads();

    scanIncBlock<OP>(ps, idx);
    scanIncBlock<OP>(negPs, idx);

    __syncthreads();

    // SORT FOR BITS
    if (glb_threadidx < N)
    {
        int len_false = negPs[block_size - 1];

        int iT, iF;
        iT = ps[idx] - 1 + len_false;
        iF = negPs[idx];

        if (p)
        {
            iT = ps[idx] - 1 + len_false;
            loc_data[iT] = dat4;
        }
        else
        {
            iF = negPs[idx];
            loc_data[iF - 1] = dat4;
        }
    }
}

// template <class OP, int block_size>
// __device__ void partition2_tiled(uint32_t *loc_data, uint32_t iteration, uint32_t bstart, uint32_t iter, uint32_t N)
// {
//     __shared__ uint32_t ps[1024];
//     __shared__ uint32_t negPs[1024];

//     const uint32_t loc_threadidx = (threadIdx.y * blockDim.x) + threadIdx.x;
//     const uint32_t glb_threadidx = getGlobalIdx();

//     const int glb_memoffset = 4 * glb_threadidx;
//     const int loc_memoffset = 4 * loc_threadidx;
//     const int blockidx = blockIdx.x + blockIdx.y * gridDim.x;

//     const uint32_t mask = (1 << (bstart + iteration));
//     uint32_t data[4];
//     uint32_t p;
//     uint32_t negP;

//     __syncthreads();
//     for (int i = 0; i < 4; i++)
//     {

//         uint32_t idx = loc_memoffset + i;

//         ps[idx] = 0;
//         negPs[idx] = 0;

//         __syncthreads();

//         if (glb_memoffset + i < N)
//         {
//             data[i] = loc_data[idx];

//             p = (data[i] & mask) >> (iteration + iter * 4);
//             negP = 1 - p;

//             ps[idx] = p;
//             negPs[idx] = negP;
//         }
//         __syncthreads();
//         //if (blockidx == 0 && iteration == 0 && iter == 0)printf(" %d\n",ps[idx]);
//     }

//     __syncthreads();

//     for (int i = 0; i < 4; i++)
//     {
//         uint32_t idx = loc_memoffset + i;

//         __syncthreads();

//         scanIncBlock<OP>(ps, idx);
//         scanIncBlock<OP>(negPs, idx);
//         __syncthreads();
//         if (blockidx == 0 && iteration == 0 && iter == 0)
//             printf("%d: %d\n", idx, ps[idx]);
//     }

//     // if (blockidx == 0 && iteration == 0 && iter == 0  && loc_threadidx == 0){
//     //     {
//     //         for (size_t i = 0; i < 1024; i++)
//     //         {
//     //             printf("%d\n",ps[i]);
//     //         }
//     //     }
//     // }

//     for (int i = 0; i < 4; i++)
//     {
//         uint32_t idx = loc_memoffset + i;

//         //if (iteration == 0 && iter == 0)printf("%d: %d\n",idx, ps[idx]);

//         __syncthreads();
//         if (glb_memoffset + i < N)
//         {
//             uint32_t len_false = negPs[block_size - 1];

//             uint32_t iT, iF;
//             iT = ps[idx] - 1 + len_false;
//             iF = negPs[idx];

//             if (p)
//             {

//                 iT = ps[idx] - 1 + len_false;
//                 //printf("P: %d: %d: %d: %d: %d\n", idx, iT, len_false, loc_threadidx,i);

//                 loc_data[iT] = data[i];
//             }
//             else
//             {
//                 iF = negPs[idx];
//                 //printf("NegP: %d: %d: %d\n", idx, iF, len_false);

//                 loc_data[iF - 1] = data[i];
//             }
//         }

//         __syncthreads();
//     }
// }

#endif // HISTO_HELPER
