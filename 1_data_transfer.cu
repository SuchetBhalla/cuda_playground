/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
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
 1. https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
 2. https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/optimize-data-transfers/bandwidthtest.cu
*/

#include <stdio.h>
#include <assert.h>

// function definitions
inline cudaError_t cudaCheck(cudaError_t result){
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

void profile_copying(float *hi, float *ho, float *d, size_t n, const char *desc){
  /*
  this function is used to compare the rate of memory transfer
  between the device and
  1. pageable memory on the host
  2. pinned memory on the host
  */

  printf("\n%s transfers\n", desc);

  // create events
  cudaEvent_t start, stop;
  cudaCheck(cudaEventCreate(&start));
  cudaCheck(cudaEventCreate(&stop));

  // copy from host to device
  size_t bytes= n *sizeof(float);
  cudaCheck(cudaEventRecord(start, 0));
  cudaCheck(cudaMemcpy(d, hi, bytes, cudaMemcpyHostToDevice));
  cudaCheck(cudaEventRecord(stop, 0));
  cudaCheck(cudaEventSynchronize(stop));
  float sec;
  cudaCheck(cudaEventElapsedTime(&sec, start, stop));
  printf("\tBandwidth (GB/sec), host to device: %.2f\n", bytes * 1e-6 / sec);

  // copy from device to host
  cudaCheck(cudaEventRecord(start, 0));
  cudaCheck(cudaMemcpy(ho, d, bytes, cudaMemcpyDeviceToHost));
  cudaCheck(cudaEventRecord(stop, 0));
  cudaCheck(cudaEventSynchronize(stop));
  cudaCheck(cudaEventElapsedTime(&sec, start, stop));
  printf("\tBandwidth (GB/sec), device to host: %.2f\n", bytes * 1e-6 / sec);

  // validate copying
  for(size_t i= 0; i < n; ++i)
    if (ho[i] != hi[i]){
      printf("*** %s transfers failed", desc);
      break;
    }

  // cleanup
  cudaCheck(cudaEventDestroy(start));
  cudaCheck(cudaEventDestroy(stop));
}

int main(){
  // init
  const size_t size= 4 * 1024 * 1024;
  const size_t bytes= size * sizeof(float);

  // arrays
  float
    *hi_pageable, *ho_pageable,
    *hi_pinned, *ho_pinned,
    *d;

  // allocate memory
  // a. host: pageable memory
  hi_pageable= (float*) malloc(bytes);
  ho_pageable= (float*) malloc(bytes);
  // b. host: pinned memory
  cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&hi_pinned), bytes));
  cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&ho_pinned), bytes));
  // c. device
  cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d), bytes));

  // initialize arrays
  for(size_t i= 0; i < size; ++i){
    hi_pageable[i]= i;
    hi_pinned[i]= i;
  }

  // display info. about device
  cudaDeviceProp prop;
  cudaCheck(cudaGetDeviceProperties(&prop, 0));
  printf("Device: %s\n", prop.name);

  printf("Transfer size (MB): %lu\n", bytes / (1<<20));

  // profile copying
  profile_copying(hi_pageable, ho_pageable, d, size, "Pageable");
  profile_copying(hi_pinned, ho_pinned, d, size, "Pinned");

    // cleanup
    cudaFree(d);
    cudaFreeHost(hi_pinned);
    cudaFreeHost(ho_pinned);
    free(hi_pageable);
    free(ho_pageable);

    return 0;
}
