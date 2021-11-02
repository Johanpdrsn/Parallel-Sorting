#ifndef KERNELS
#define KERNELS

#include "cub.cuh"

// Give cred to the cheatsheet
// Gets global threadId of 2D grid of 2D blocks
__device__ int getGlobalIdx(){
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int threadId = blockId * (blockDim.x * blockDim.y)
    + (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}

//Unstable countSort - geeksforgeeks implementation
void countSort(char arr[]){
  char output[strlen(arr)];
 
  int count[16], i;
  memset(count, 0, sizeof(count));
 
  for (i = 0; arr[i]; ++i)
    ++count[arr[i]];
 
  for (i = 1; i <= 16; ++i)
    count[i] += count[i - 1];
 
  for (i = 0; arr[i]; ++i) {
    output[count[arr[i]] - 1] = arr[i];
    --count[arr[i]];
  }
 
  for (i = 0; arr[i]; ++i)
    arr[i] = output[i];
}

template <int N> //<class ElTp> <- we will need this to generalize 
__global__ void kern1(uint32_t* data_keys_in
                      , uint32_t* data_keys_out
                      , uint32_t* glb_bins
		      , int iter
                        ) {
  __shared__ uint32_t loc_data[N];
  __shared__ uint32_t bins[16];

  uint32_t data[4];

  //memset here?

  int blockidx = blockIdx.x + blockIdx.y * gridDim.x;
  int glb_threadidx = getGlobalIdx();
  int loc_threadidx = (threadIdx.y * blockDim.x) + threadIdx.x;

  // This is not coalesced on memory (we should stride instead of taking 4 seq)
  int glb_memoffset = 4 * glb_threadidx;
  int loc_memoffset = 4 * loc_threadidx;
  // Loop over 4 data entries
  for(int i=0; i<4; i++){
    // if  glb_memoffset < N - ??
    data[i] = data_keys_in[glb_memoffset + i];
    loc_data[loc_memoffset + i] = data[i];
  }

  /* is this coalesced?
  int glb_memoffset = glb_threadidx;
  int loc_memoffset = loc_threadidx;
  for(int i=0; i<4; i++){
    data[i] = data_keys_in[i*glb_memoffset];
    loc_data[loc_memoffset*i] = data[i];
  }*/

  
  __syncthreads();

  int bstart = iter *  4;
  int binidx;
  uint32_t mask;
  mask = ((1 << 4) << bstart);
  // Loop over 4 data entries
  for(int i=0; i<4; i++){
    binidx = data[i] & mask;
    atomicAdd(&bins[binidx], 1);
  }

  __syncthreads();
  
  // THIS IS WRONG - THE WHOLE SHARED DATA NEEDS TO BE SORTED HERE
  //countSort(data);

  for(int i=0; i<4; i++){
    // if  glb_memoffset < N - ??
    loc_data[loc_memoffset + i] = data[i];
  }

  

  //sort local data using counting sort


}
#endif 
