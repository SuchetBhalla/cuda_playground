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
1.https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
2. https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/overlap-data-transfers/async.cu
 */

 #include <stdio.h>

// kernel
__global__ void increment(float *d, int offset){
  int idx= threadIdx.x + blockIdx.x * blockDim.x + offset;
  float
    x= static_cast<float>(idx),
    s= sinf(x),
    c= cosf(x);
  d[idx]= d[idx] + sqrtf(s*s + c*c); // add 1.0
}

// function definitions
float maxError(float *h, int n){
  float Err= 0, err;
  for(int i= 0; i < n; ++i){
    err= fabs(h[i] - 1.0f);
    if (err > Err)
      Err= err;
  }
  return Err;
}

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

int main(int argc, char **argv){

  // 1. init
  int devId= 0;
  if (argc > 1)
    devId= atoi(argv[1]);
  cudaCheck(cudaSetDevice(devId));

  // display device's properties
  cudaDeviceProp prop;
  cudaCheck(cudaGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  if (prop.asyncEngineCount > 0)
    printf("Concurrency is possible. Engine count: %d\n", prop.asyncEngineCount);
  else
    printf("Concurrency is not possible.\n");

  // allocate memory, create streams & events
  const int
    blockSize= 256,
    nStreams= 4,
    size= nStreams * blockSize * 1024 * 4,
    bytes= size * sizeof(float),
    streamSize= size / nStreams,
    streamBytes= streamSize * sizeof(float);

float *h, *d;
cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&h), bytes)); // pinned Memory
cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d), bytes));

cudaEvent_t startEvent, stopEvent;
cudaCheck(cudaEventCreate(&startEvent));
cudaCheck(cudaEventCreate(&stopEvent));

cudaStream_t stream[nStreams];
for (int i= 0; i < nStreams; ++i)
  cudaCheck(cudaStreamCreate(&stream[i]));

// 2. main

// naive: operations launched sequentially in the default stream
memset(h, 0, bytes);
cudaCheck(cudaEventRecord(startEvent, 0));
cudaCheck(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
increment<<<streamSize/blockSize, blockSize>>>(d, 0);
cudaCheck(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));
cudaCheck(cudaEventRecord(stopEvent, 0));
cudaCheck(cudaEventSynchronize(stopEvent));
float msec;
cudaCheck(cudaEventElapsedTime(&msec, startEvent, stopEvent));
printf("Time for synchronous data-transfer & execution: %.3f msec\n", msec);
printf("Max. error: %.1f\n", maxError(h, size));

/*
concurrent version 1
loop over:
  1. data-transfer; host to device
  2. computation
  3. data-transfer; device to host
*/
cudaCheck(cudaEventRecord(startEvent, 0));
for(int i=0, offset= 0, gridSize= streamSize/blockSize; i < nStreams; ++i, offset+=streamSize){
  // 1
  cudaCheck(cudaMemcpyAsync(d+offset, h+offset, streamBytes, cudaMemcpyHostToDevice, stream[i]));
  // 2
  increment<<<gridSize, blockSize, 0, stream[i]>>>(d, offset); // # threads ==  streamSize
  // 3
  cudaCheck(cudaMemcpyAsync(h+offset, d+offset, streamBytes, cudaMemcpyDeviceToHost, stream[i]));
}
cudaCheck(cudaEventRecord(stopEvent, 0));
cudaCheck(cudaEventSynchronize(stopEvent));
cudaCheck(cudaEventElapsedTime(&msec, startEvent, stopEvent));
printf("Time for asynchronous (v1) data-transfer & execution: %.3f msec\n", msec);
printf("Max. error: %.1f\n", maxError(h, size));

/*
concurrent version 2
loop over data-transfer; host to device
loop over computation
loop over data-transfer; device to host
*/
cudaCheck(cudaEventRecord(startEvent, 0));
// 1
for(int i=0, offset= 0; i < nStreams; ++i, offset+=streamSize)
  cudaCheck(cudaMemcpyAsync(d+offset, h+offset, streamBytes, cudaMemcpyHostToDevice, stream[i]));
// 2
for(int i=0, offset= 0, gridSize= nStreams/blockSize; i < nStreams; ++i, offset+=streamSize)
  increment<<<gridSize, blockSize, 0, stream[i]>>>(d, offset);
// 3
for(int i=0, offset= 0; i < nStreams; ++i, offset+=streamSize)
  cudaCheck(cudaMemcpyAsync(h+offset, d+offset, streamBytes, cudaMemcpyDeviceToHost, stream[i]));

cudaCheck(cudaEventRecord(stopEvent, 0));
cudaCheck(cudaEventSynchronize(stopEvent));
cudaCheck(cudaEventElapsedTime(&msec, startEvent, stopEvent));
printf("Time for asynchronous (v2) data-transfer & execution: %.3f msec\n", msec);
printf("Max. error: %.1f\n", maxError(h, size));

// 3. cleanup
for(int i= 0; i < nStreams; ++i)
  cudaCheck(cudaStreamDestroy(stream[i]));
cudaCheck(cudaEventDestroy(stopEvent));
cudaCheck(cudaEventDestroy(startEvent));
cudaCheck(cudaFree(d));
cudaCheck(cudaFreeHost(h));

return 0;
}
