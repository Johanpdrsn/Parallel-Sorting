#ifndef KERNELS
#define KERNELS

#include "cub.cuh"

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

 
    // This is not coalesced on memory (we should stride instead of taking 4 seq)
    const int glb_memoffset = 4 * glb_threadidx;
    const int loc_memoffset = 4 * loc_threadidx;
    // Read data from global memory.
    // Loop over 4 data entries.
    for (int i = 0; i < 4; i++)
    {
        // if  glb_memoffset < N - ??
        if (((glb_memoffset + i) <= N) && ((loc_memoffset + i) <= block_size)){
            data[i] = data_keys_in[glb_memoffset + i];
            loc_data[loc_memoffset + i] = data[i];
        }
    }

    /* is this coalesced?
  int glb_memoffset = glb_threadidx;
  int loc_memoffset = loc_threadidx;
  for(int i=0; i<4; i++){
    data[i] = data_keys_in[i*glb_memoffset];
    loc_data[loc_memoffset*i] = data[i];
  }*/

    // Sync needed after loading all data in.
    __syncthreads();

    int bstart = iter * 4;
    int binidx;
    
    uint32_t mask;
    mask = (15 << bstart);
    // Loop over 4 data entries
    for (int i = 0; i < 4; i++)
    {
        binidx = (data[i] & mask) >> bstart;
        int old = atomicAdd(&local_histogram[binidx], 1);

        ranksInBins[i] = old;
        binsForElms[i] = binidx;
    }

    // Need to sync, cannot sort before the histogram is done.
    __syncthreads();


    for (size_t i=0; i<16; i++){
        if (i==0){
            scan_local_histogram[i] = 0;
        }
        else{
            scan_local_histogram[i] = local_histogram[i-1] + scan_local_histogram[i-1];
        }
    }
    
    __syncthreads();

    // SORT LOCAL TILE
    for (int i = 0; i < 4; i++)
    {
        if  ((loc_memoffset + i) < block_size){
            uint idx = scan_local_histogram[binsForElms[i]] + ranksInBins[i];
            loc_data[idx] = data[i];
        }
    }

    __syncthreads();

    // WRITE LOCAL HISTOGRAMS TO GLOBAL
    if (loc_threadidx == 0){
        uint32_t p = blockDim.x * blockDim.y;
        printf("%d\n",p);
        for (size_t i = 0; i < 16; i++)
        {
            glb_histogram[p*i+blockidx] = local_histogram[i];
        }
    }




    // WRITE SORTED TILE TO GLOBAL
    // for (int i = 0; i < 4; i++)
    //     {
    //         if (((glb_memoffset + i) <= N) && ((loc_memoffset + i) <= block_size)){
    //             data_keys_in[glb_memoffset + i] = loc_data[loc_memoffset+i];
    //         }
    //     }




}
#endif
