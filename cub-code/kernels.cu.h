#ifndef KERNELS
#define KERNELS

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

//Unstable countSort - geeksforgeeks implementation
void countSort(char arr[])
{
    char output[strlen(arr)];

    int count[16], i;
    memset(count, 0, sizeof(count));

    for (i = 0; arr[i]; ++i)
        ++count[arr[i]];

    for (i = 1; i <= 16; ++i)
        count[i] += count[i - 1];

    for (i = 0; arr[i]; ++i)
    {
        output[count[arr[i]] - 1] = arr[i];
        --count[arr[i]];
    }

    for (i = 0; arr[i]; ++i)
        arr[i] = output[i];
}

template <int block_size> //<class ElTp> <- we will need this to generalize
__global__ void kern1(uint32_t *data_keys_in, uint32_t *data_keys_out, uint32_t *glb_histogram, int N, int iter)
{
    __shared__ uint32_t loc_data[block_size];
    __shared__ uint32_t local_histogram[16];

    uint32_t scan_local_histogram[16];

    uint32_t data[4];
    uint32_t binsForElms[4];
    uint32_t ranksInBins[4];

    //memset here?

    const int blockidx = blockIdx.x + blockIdx.y * gridDim.x;
    const int glb_threadidx = getGlobalIdx();
    const int loc_threadidx = (threadIdx.y * blockDim.x) + threadIdx.x;

    if (loc_threadidx < 16)
        local_histogram[loc_threadidx] = 0;

    // This is not coalesced on memory (we should stride instead of taking 4 seq)
    const int glb_memoffset = 4 * glb_threadidx;
    const int loc_memoffset = 4 * loc_threadidx;
    // Read data from global memory.
    // Loop over 4 data entries.
    for (int i = 0; i < 4; i++)
    {
        if ((glb_memoffset + i) < N)
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
        if ((glb_memoffset + i) < N)
        {
            binidx = (data[i] & mask) >> bstart;
            int old = atomicAdd(&local_histogram[binidx], 1);

            ranksInBins[i] = old;
            binsForElms[i] = binidx;
        }
    }

    // Need to sync, cannot sort before the histogram is done.
    __syncthreads();

    // SCAN LOCAL HISTOGRAM
    for (size_t i = 0; i < 16; i++)
    {
        if (i == 0)
        {
            scan_local_histogram[i] = 0;
        }
        else
        {
            scan_local_histogram[i] = local_histogram[i-1] + scan_local_histogram[i - 1];
        }
    }

    __syncthreads();

    // SORT LOCAL TILE
    for (int i = 0; i < 4; i++)
    {
        if ((glb_memoffset + i) < N)
        {
            uint32_t idx = scan_local_histogram[binsForElms[i]] + ranksInBins[i];
            loc_data[idx] = data[i];
        }
    }



    __syncthreads();

    // WRITE LOCAL HISTOGRAMS TO GLOBAL
    if (loc_threadidx == 0)
    {
        uint32_t p = gridDim.x * gridDim.y;
        for (size_t i = 0; i < 16; i++)
        {
            glb_histogram[p * i + blockidx] = local_histogram[i];
        }
    }

    // // WRITE SORTED TILE TO GLOBAL
    for (int i = 0; i < 4; i++)
    {
        if ((glb_memoffset + i) < N)
        {
            data_keys_out[glb_memoffset + i] = loc_data[loc_memoffset + i];
        }
    }



}

template <int block_size> //<class ElTp> <- we will need this to generalize
__global__ void kern3(uint32_t *glb_histogram_in, uint32_t *glb_histogram_out, int N)
{
    typedef cub::BlockScan<int, block_size> BlockScan;
    typedef cub::BlockLoad<int, block_size, 16> BlockLoad;

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    const int glb_threadidx = getGlobalIdx();

    __shared__ typename BlockScan::TempStorage scan_temp_storage;
    __shared__ typename BlockLoad::TempStorage load_temp_storage;

    int thread_data[16];

    BlockLoad(load_temp_storage).Load(glb_histogram_in, thread_data);

    __syncthreads();

    BlockScan(scan_temp_storage).ExclusiveSum(thread_data, thread_data);

    __syncthreads();

    for (size_t i = 0; i < 16; i++)
    {
        glb_histogram_out[blockId * 16 + i] = thread_data[i];
    }

}

template <int block_size> //<class ElTp> <- we will need this to generalize
__global__ void kern4(uint32_t *glb_histogram_in, uint32_t *glb_data_in, uint32_t *glb_data_out ,int N, int iter, uint32_t* hist)
{

    __shared__ uint32_t loc_data[block_size];
    __shared__ uint32_t local_histogram[16];

    uint32_t scan_local_histogram[16];

    uint32_t data[4];

    const int blockidx = blockIdx.x + blockIdx.y * gridDim.x;
    const int glb_threadidx = getGlobalIdx();
    const int loc_threadidx = (threadIdx.y * blockDim.x) + threadIdx.x;

    if (loc_threadidx < 16)
        local_histogram[loc_threadidx] = 0;

    // This is not coalesced on memory (we should stride instead of taking 4 seq)
    const int glb_memoffset = 4 * glb_threadidx;
    const int loc_memoffset = 4 * loc_threadidx;
    const int p = gridDim.x * gridDim.y;

    int32_t elm;
    int32_t elmBin;
    int32_t glbScanElm;
    int32_t loc_scan_offset;
    // Read data from global memory.
    // Loop over 4 data entries.
    for (int i = 0; i < 4; i++)
    {
        if ((glb_memoffset + i) < N)
	  {
	    
	    data[i] = glb_data_in[glb_memoffset + i];
	    loc_data[loc_memoffset + i] = data[i];
	  }
    }

    __syncthreads();

    //local hist
    for (int i = 0; i < 16; i++)
    { 
        local_histogram[i] = hist[p*i+blockidx];
    }

    // scanned local hist
    for (size_t i = 0; i < 16; i++)
    {
        if (i == 0)
        {
            scan_local_histogram[i] = 0;
        }
        else
        {
            scan_local_histogram[i] = local_histogram[i-1] + scan_local_histogram[i - 1];
        }
    }

    
    __syncthreads();

    for (int i = 0; i < 4; i++)
    {
        if ((glb_memoffset + i) < N)
	    {
            
            elm = loc_data[loc_threadidx*4 + i]; //V
            // binOf function
            int bstart = iter * 4;
            uint32_t mask;
            mask = (15 << bstart); 

            elmBin = (data[i] & mask) >> bstart;   //V
            glbScanElm = glb_histogram_in[p * elmBin + blockidx];//V
            if (elmBin == 0){
                loc_scan_offset = 0; 
            } else{
                loc_scan_offset = scan_local_histogram[elmBin]; 
            }
            
            glb_data_out[glbScanElm + (loc_threadidx*4 + i - loc_scan_offset)] = elm;
	    }

    }

    // if (loc_threadidx == 0 && blockidx == 0){
    //     for (int i = 0; i < block_size; i++)
    //     {
    //         printf("%d\n", loc_data[i]);
    //     }  
    // }
}

#endif
