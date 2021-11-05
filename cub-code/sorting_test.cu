//#include "../../cub-1.8.0/cub/cub.cuh"   // or equivalently <cub/device/device_histogram.cuh>
#include "cub.cuh"
#include "helper.cu.h"
#include "kernels.cu.h"
#include <math.h>

#define blockMemSize 1024

void printBits(int val){
    for(unsigned int mask = 0x80000000; mask; mask >>= 1){
         printf("%d", !!(mask & val));
    }
    printf("\n");
}

template<class Z>
bool validateZ(Z* A, uint32_t sizeAB) {
    for(uint32_t i = 1; i < sizeAB; i++)
      if (A[i-1] > A[i]){
        printf("INVALID RESULT for i:%d, (A[i-1]=%d > A[i]=%d)\n", i, A[i-1], A[i]);
        return false;
      }
    return true;
}

void randomInitNat(uint32_t* data, const uint32_t size, const uint32_t H) {
    for (int i = 0; i < size; ++i) {
        unsigned long int r = rand()%16;
        data[i] = r % H;
        
    }
}

double sortRedByKeyCUB( uint32_t* data_keys_in
                      , uint32_t* data_keys_out
                      , const uint64_t N
) {
    int beg_bit = 0;
    int end_bit = 32;

    void * tmp_sort_mem = NULL;
    size_t tmp_sort_len = 0;

    { // sort prelude
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
                                      );
        cudaMalloc(&tmp_sort_mem, tmp_sort_len);
    }
    cudaCheckError();

    { // one dry run
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
                                      );
        cudaDeviceSynchronize();
    }
    cudaCheckError();

    // timing
    double elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for(int k=0; k<GPU_RUNS; k++) {
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
                                      );
    }
    cudaDeviceSynchronize();
    cudaCheckError();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / ((double)GPU_RUNS);

    cudaFree(tmp_sort_mem);

    return elapsed;
}


int main (int argc, char * argv[]) {
    if (argc != 2) {
        printf("Usage: %s <size-of-array>\n", argv[0]);
        exit(1);
    }
    const uint64_t N = atoi(argv[1]);

    //Allocate and Initialize Host data with random values
    uint32_t* h_keys  = (uint32_t*) malloc(N*sizeof(uint32_t));
    uint32_t* h_keys_res  = (uint32_t*) malloc(N*sizeof(uint32_t));
    randomInitNat(h_keys, N, N/10);

    //Allocate and Initialize Device data
    uint32_t* d_keys_in;
    uint32_t* d_keys_out;
    cudaSucceeded(cudaMalloc((void**) &d_keys_in,  N * sizeof(uint32_t)));
    cudaSucceeded(cudaMemcpy(d_keys_in, h_keys, N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    cudaSucceeded(cudaMalloc((void**) &d_keys_out, N * sizeof(uint32_t)));

    double elapsed = sortRedByKeyCUB( d_keys_in, d_keys_out, N );

    cudaMemcpy(h_keys_res, d_keys_out, N*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaCheckError();

    bool success = validateZ(h_keys_res, N);

    printf("CUB Sorting for N=%lu runs in: %.2f us, VALID: %d\n", N, elapsed, success);

    // Cleanup and closing
    cudaFree(d_keys_in); cudaFree(d_keys_out);
    free(h_keys); free(h_keys_res);


    //  ** New kernel section ** 
    // setup execution parameters
    int dimbl = 1;//(int) (sqrt(ceil(N/1024))) + 1;
    dim3 block(32, 32, 1); // 256 threads per block
    dim3 grid (dimbl, dimbl, 1); 

    //Allocate and Initialize Host data with random values
    
    uint32_t* global_histogram_output  = (uint32_t*) malloc(dimbl * dimbl * 16 *sizeof(uint32_t)); // todo:fix size, but who cares
    uint32_t* keys  = (uint32_t*) malloc(N*sizeof(uint32_t));
    uint32_t* keys_res  = (uint32_t*) malloc(N*sizeof(uint32_t));
    randomInitNat(keys, N, N/10);

    //Allocate and Initialize Device data
    uint32_t* keys_in;
    uint32_t* keys_sort;
    uint32_t* keys_out;
    uint32_t* glb_bins;
    uint32_t* scanned_glb_bins;

    uint32_t num_glb_bins = dimbl * dimbl * 16 * sizeof(uint32_t);
    cudaSucceeded(cudaMalloc((void**) &keys_in,  N * sizeof(uint32_t)));
    cudaSucceeded(cudaMemcpy(keys_in, keys, N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    cudaSucceeded(cudaMalloc((void**) &keys_sort,  N * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &keys_out, N * sizeof(uint32_t)));

    cudaSucceeded(cudaMalloc((void**) &glb_bins, num_glb_bins));
    cudaSucceeded(cudaMalloc((void**) &scanned_glb_bins, num_glb_bins));
    cudaMemset(glb_bins, 0, dimbl * dimbl * 16 * sizeof(uint32_t));
    cudaMemset(scanned_glb_bins, 0, dimbl * dimbl * 16 * sizeof(uint32_t));

    //    double elapsed = sortRedByKeyCUB( keys_in, deys_out, N );
    double elapsedKernel;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    // Initialize vars for devicescan
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, glb_bins, scanned_glb_bins, num_glb_bins);  
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    for(int q=0; q<GPU_RUNS; q++) {
        for (int iter=0; iter<1; iter++){
            kern1<blockMemSize><<< grid, block >>>(keys_in, keys_out, glb_bins, N ,iter);
            //kern3<blockMemSize><<< grid, block >>>(glb_bins, scanned_glb_bins, num_glb_bins);
            //cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, glb_bins, scanned_glb_bins, num_glb_bins);	
            //kern4<blockMemSize><<< grid, block >>>(scanned_glb_bins, keys_out, keys_sort, N ,iter, glb_bins);
            //keys_in = keys_sort;
        }
    }
    cudaDeviceSynchronize();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsedKernel = (t_diff.tv_sec*1e6+t_diff.tv_usec) / ((double)GPU_RUNS);


    cudaMemcpy(keys_res, keys_out, N*sizeof(uint32_t), cudaMemcpyDeviceToHost); // todo: fix keys_in
    cudaDeviceSynchronize();
    cudaCheckError();
    
    
    // for (size_t i = 0; i < N; i++)
    //   {
	//     printf("%d\n",keys_res[i]);
    //     //printBits(keys_res[i]);
    //   }
    
    cudaMemcpy(global_histogram_output, glb_bins, dimbl * dimbl* 16 *sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaCheckError();
    


    bool successKernel = validateZ(keys_res, N);

    printf("Our sorting for N=%lu runs in: %.2f us, VALID: %d\n", N, elapsedKernel, successKernel);

    // Cleanup and closing
    cudaFree(keys_in); cudaFree(keys_out); cudaFree(keys_sort); cudaFree(glb_bins); cudaFree(scanned_glb_bins);
    cudaFree(d_temp_storage);
    free(keys); free(keys_res);


    return success ? 0 : 1;

}
