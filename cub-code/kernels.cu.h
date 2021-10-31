#ifndef KERNELS
#define KERNELS

// Give cred to the cheatsheet
// Gets global threadId of 2D grid of 2D blocks
__device__ int getGlobalIdx(){
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int threadId = blockId * (blockDim.x * blockDim.y)
    + (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}

template <int N> //<class ElTp> <- we will need this to generalize 
__global__ void kern1(uint32_t* data_keys_in
                        , uint32_t* data_keys_out
                        ) {
  __shared__ uint32_t loc_data[N];
  __shared__ uint32_t bins[16];

  uint32_t data[4];

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



}
#endif 
