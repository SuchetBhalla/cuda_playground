/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 ** Redistributions of source code must retain the above copyright
 *  notice, this list of conditions and the following disclaimer.
 ** Redistributions in binary form must reproduce the above copyright
 *  notice, this list of conditions and the following disclaimer in the
 *  documentation and/or other materials provided with the distribution.
 ** Neither the name of NVIDIA CORPORATION nor the names of its
 *  contributors may be used to endorse or promote products derived
 *  from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
references:
1. https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
2. https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu
*/

#include <stdio.h>
#include <assert.h>

// global variables (but local to this translation unit)
static const int
  COLS = 1024,
  ROWS= 1024,
  TILE_DIM= 32,
  BLOCK_ROWS= 8,
  NUM_REPS= 100;

// kernels
__global__ void copy(float *op, const float *ip){

  // indices into a tile of dimension (TILE_DIM, TILE_DIM, 1)
  int
    x= threadIdx.x + blockIdx.x * TILE_DIM,
    y= threadIdx.y + blockIdx.y * TILE_DIM,
    width= COLS;

  /*
  0. HBM -> HBM
  1. Each thread is used to copy multiple elements.
  Q. How many?
  A. TILE_DIM / BLOCK_ROWS = 32 / 8 = 4
  2. j = stride into the tile
  - here, j \in (0, 8, 16, 24)
  3. In a warp,
  - threadIdx.x varies; \in [0, 31]
  - threadIdx.y is constant
  - consequentially, 32 contiguous floats are read from & written to global memory
  i.e., coalesced read & write
  */
  for (int j= 0; j < TILE_DIM; j+=BLOCK_ROWS)
    op[x + (y+j)*width]= ip[x + (y+j)*width];
}

__global__ void copysram(float *op, const float *ip){

// a buffer shared amongst the threads of a block
__shared__ float buffer[TILE_DIM * TILE_DIM];

// indices into a tile of dimension (TILE_DIM, TILE_DIM, 1)
int
    x= threadIdx.x + blockIdx.x * TILE_DIM,
    y= threadIdx.y + blockIdx.y * TILE_DIM,
    width= COLS;

// HBM -> SRAM
for(int j= 0; j < TILE_DIM; j+=BLOCK_ROWS)
  buffer[threadIdx.x + (threadIdx.y+j)*TILE_DIM]= ip[x + (y+j)*width];

// "barrier" for all threads in a block
__syncthreads();

// SRAM -> HBM
for(int j= 0; j < TILE_DIM; j+=BLOCK_ROWS)
  op[x + (y+j)*width]= buffer[threadIdx.x + (threadIdx.y+j)*TILE_DIM];

}

__global__ void transposeNaive(float *op, const float *ip){
  int
    x= threadIdx.x + blockIdx.x * TILE_DIM,
    y= threadIdx.y + blockIdx.y * TILE_DIM,
    width= COLS;

  /*
  HBM -> HBM
  coalesced read, but uncoalesced writes (totalling 32 per warp)
  Q. why?
  A. The addresses of read flosts are contiguous, but those of written floats
  are apart by width * sizeof(float), i.e., 1024 * 4
  */
  for(int j= 0; j < TILE_DIM; j+=BLOCK_ROWS)
    op[x*width + y+j]= ip[x + (y+j)*width];
}

__global__ void transposeSubOptima(float *op, const float *ip){
  __shared__ float buffer[TILE_DIM][TILE_DIM];

  int
    x= threadIdx.x + blockIdx.x * TILE_DIM,
    y= threadIdx.y + blockIdx.y * TILE_DIM,
    width= COLS;

  // HBM -> SRAM; coalesced read
  for(int j= 0; j < TILE_DIM; j+=BLOCK_ROWS)
    buffer[threadIdx.y+j][threadIdx.x]= ip[x + (y+j)*width];

  // "barrier" for all threads in a block
  __syncthreads();

  /*
  1. SRAM -> HBM; cloaesced write
  2. but, the read operation (from the buffer) is naive. A read can be completed in 1 cycle,
  but here it requires 32 cycles; due to *bank conflicts*, which serializes the (32) reads (per warp) from SRAM

  elaboration:
  a. The (entire) SRAM is split into 32 banks (== no. of threads in a warp)
  each bank can transfer 4 bytes per cycle (to a thread)

  bank idx 0  is used to access addresses 0, 128, 256, ...
  bank idx 1 <- 4, 132, 260, ...
  bank idx 2 <- 8, 136, 264, ...
  ...
  bank idx 31 <- 124, 252, 380 ...

  formula: bank idx = (address / 4) % 32

  b. when accessing a column in the buffer,
    address of each element =  buffer + {threadIdx.x * TILE_DIM+ (threadIdx.y + j)} * sizeof(float)

  for a warp,
  - threadIdx.y = constant
  - threadIdx.x \in [0, 31]
  here, TILE_DIM= 32
    => bank idx = (buffer + {threadIdx.x*32 + const}*4)/4 % 32
      = const + [0, 31] * 8 * 4 % 32
      => 32 threads map to the same bank index (though the addresses are different)
  ~ if the address was the same, then a broadcast would have occured (in 1 cycle)
  Consequently, reading a column (from a buffer) requires 32 cycles; instead of 1
  */
  x= threadIdx.x + blockIdx.y * TILE_DIM;
  y= threadIdx.y + blockIdx.x * TILE_DIM;
  for(int j= 0; j < TILE_DIM; j+=BLOCK_ROWS)
    op[x + (y+j)*width]= buffer[threadIdx.x][threadIdx.y+j]; // col -> row
}

__global__ void transposeOptima(float *op, const float *ip){

/*
the extra column (aka padding) helps prevent bank conflicts
facts:
1. buffer has 32 rows x 33 columns
2. SRAM is split into 32 banks

inference: the addresses of consecutive elements in a column are offset by 32*4 + 4
in the buffer (instead of 32*4), thus no bank conflicts.
*/
  __shared__ float buffer[TILE_DIM][TILE_DIM+1];

  int
    x= threadIdx.x + blockIdx.x * TILE_DIM,
    y= threadIdx.y + blockIdx.y * TILE_DIM,
    width= COLS;

  // HBM -> SRAM; coalesced read`
  for(int j= 0; j < TILE_DIM; j+=BLOCK_ROWS)
    buffer[threadIdx.y+j][threadIdx.x]= ip[x + (y+j)*width];

  // "barrier" for all threads in a block
  __syncthreads();

  x= threadIdx.x + blockIdx.y * TILE_DIM;
  y= threadIdx.y + blockIdx.x * TILE_DIM;

  // SRAM -> HBM; coalesced write; no bank conflicts
  for(int j= 0; j < TILE_DIM; j+=BLOCK_ROWS)
    op[x + (y+j)*width]= buffer[threadIdx.x][threadIdx.y + j];
}

// function definitions
inline cudaError_t checkCuda(cudaError_t result) {
  /*
  desc.:
  1. used to check whether the call to a cuda api failed.
  - The api must return a 'cudaError_t' (i.e., an integer)
  2. This check occurs only if debugging is enabled during compilation
  refer: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038
  */
  #if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess){
      fprintf(stderr, "CUDA runtime error: %s\n", cudaGetErrorString(result));
      assert(result == cudaSuccess);
  }
  #endif
  return result;
}

void validate(const float *refer, const float *result, int n, float ms){
  bool passed= true;
  for(int i= 0; i < n; i++)
    if (refer[i] != result[i]){
      printf("%d %f %f\n", i, refer[i], result[i]);
      printf("%25s\n", "*** FAILED ***");
      passed= false;
      break;
    }
  //2* ; for read & write
  if (passed)
    printf("%20.2f\n", 2*n*sizeof(float)*1e-6*NUM_REPS/ms);
}

int main(int argc, char **argv){
  // error checks
  if (COLS % TILE_DIM || ROWS % TILE_DIM){
    printf("COLS & ROWS must be a multiple of TILE_DIM\n");
    return 1;
  }
  if (TILE_DIM % BLOCK_ROWS){
    printf("TILE_DIM must be a multiple of BLOCK_ROWS\n");
    return 1;
  }

  // to select a specific GPU using the shell
  int devId= 0;
  if (argc > 1)
    devId= atoi(argv[1]);
  checkCuda(cudaSetDevice(devId));

  // execution configuration
  dim3 dimGrid(COLS/TILE_DIM, ROWS/TILE_DIM, 1);
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

  // print configuration
  cudaDeviceProp prop;
  checkCuda(cudaGetDeviceProperties(&prop, devId));
  printf("Device: %s\n", prop.name);
  printf("Matrix size: %d x %d, Tile size: %d x %d x %d, Block size: %d x %d x %d\n",
    ROWS, COLS, dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

  // initializations
  const int size= COLS * ROWS;
  const int mem_size= size * sizeof(float);

  // 1.a. allocate memory on host
  float
    *hidata= (float*) malloc(mem_size),
    *hodata= (float*) malloc(mem_size),
    *hidata_transposed= (float*) malloc(mem_size);

  // b. allocate memory on device
  float
    *didata, *dodata;
    checkCuda(cudaMalloc(&didata, mem_size));
    checkCuda(cudaMalloc(&dodata, mem_size));

  // 2. initialize arrays

  // a. host
  for(int i= 0; i < ROWS; ++i){
    int stride= i*COLS;
    for(int j= 0, idx; j < COLS; ++j){
      idx= j + stride;
      hidata[idx]= idx;
      hidata_transposed[idx]= j*COLS + i;
    }
  }

  // b. device
  checkCuda(cudaMemcpy(didata, hidata, mem_size, cudaMemcpyHostToDevice));

  // 3. create events, to track time for computation
  float sec;
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));

  // main
  printf("%25s%25s\n", "Routine", "Bandwidth (GB/sec)");
  printf("%25s", "copy");

  // copy
  copy<<<dimGrid, dimBlock>>>(dodata, didata); // warm up
  checkCuda(cudaEventRecord(startEvent, 0));
  for(int i= 0; i < NUM_REPS; ++i)
    copy<<<dimGrid, dimBlock>>>(dodata, didata);
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent)); // host must wait for the device to complete this kernel's execution
  checkCuda(cudaEventElapsedTime(&sec, startEvent, stopEvent));
  checkCuda(cudaMemcpy(hodata, dodata, mem_size, cudaMemcpyDeviceToHost));
  validate(hidata, hodata, size, sec);

  // copy SRAM
  printf("%25s", "copysram");
  copysram<<<dimGrid, dimBlock>>>(dodata, didata); // warm up
  checkCuda(cudaEventRecord(startEvent, 0));
  for(int i= 0; i < NUM_REPS; ++i)
    copysram<<<dimGrid, dimBlock>>>(dodata, didata);
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&sec, startEvent, stopEvent));
  checkCuda(cudaMemcpy(hodata, dodata, mem_size, cudaMemcpyDeviceToHost));
  validate(hidata, hodata, size, sec);

  // transpose naive
  printf("%25s", "transposeNaive");
  transposeNaive<<<dimGrid, dimBlock>>>(dodata, didata); // warm up
  checkCuda(cudaEventRecord(startEvent, 0));
  for(int i= 0; i < NUM_REPS; ++i)
    transposeNaive<<<dimGrid, dimBlock>>>(dodata, didata);
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&sec, startEvent, stopEvent));
  checkCuda(cudaMemcpy(hodata, dodata, mem_size, cudaMemcpyDeviceToHost));
  validate(hidata_transposed, hodata, size, sec);

  // transposeSubOptima
  printf("%25s", "transposeSubOptima");
  transposeSubOptima<<<dimGrid, dimBlock>>>(dodata, didata); // warm up
  checkCuda(cudaEventRecord(startEvent, 0));
  for(int i= 0; i < NUM_REPS; ++i)
    transposeSubOptima<<<dimGrid, dimBlock>>>(dodata, didata);
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&sec, startEvent, stopEvent));
  checkCuda(cudaMemcpy(hodata, dodata, mem_size, cudaMemcpyDeviceToHost));
  validate(hidata_transposed, hodata, size, sec);

  // transposeOptima
  printf("%25s", "transposeOptima");
  transposeOptima<<<dimGrid, dimBlock>>>(dodata, didata); // warm up
  checkCuda(cudaEventRecord(startEvent, 0));
  for(int i= 0; i < NUM_REPS; ++i)
    transposeOptima<<<dimGrid, dimBlock>>>(dodata, didata);
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&sec, startEvent, stopEvent));
  checkCuda(cudaMemcpy(hodata, dodata, mem_size, cudaMemcpyDeviceToHost));
  validate(hidata_transposed, hodata, size, sec);

  // cleanup
  checkCuda(cudaEventDestroy(stopEvent));
  checkCuda(cudaEventDestroy(startEvent));
  checkCuda(cudaFree(dodata));
  checkCuda(cudaFree(didata));
  free(hodata);
  free(hidata);

  return 0;
}
