// refer: https://developer.nvidina.com/blog/cuda-graphs/

/*
The objective of this program is showcase that the latency of communication
between the host & device, can be significant in comparision to the time spent
on execution.
*/

#include <algorithm> // max
#include <iostream>
#include <chrono>
#include <cstdint> // int_fast16_t
#include <cmath> // fabs

using namespace std;

// 'constexpr' guarantees compile-time evaluation
constexpr size_t SIZE= 1024 * 1024; // ~1 Million elements
/*
facts:
1. A T4 has 40 SMs.
2. Each SM can run 1024 threads, parallely.

deduction: If I must run 4 blocks per SM, then the number of threads must equal 1024
=> 1024/4 = 256 threads per block
*/
constexpr size_t
  BLOCKS= 4 * 40,
  THREADS= 256;

// no. of iterations== STEPS * NSYNC
constexpr size_t
  STEPS= 1000,
  NSYNC= 50;

// kernels
__global__
void initializeArray(float *darr, float val, int n){
  // desc. set each element of 'darr' equal to val

  // returns a thread's index withinn a grid
  int
    i= threadIdx.x + blockIdx.x * blockDim.x, // [0, 31] + [0, 159] * 256
    stride= blockDim.x * gridDim.x; // 256 * 160

  for(int idx= i; idx < n; idx+=stride)
    darr[idx]= val;
}

__global__
void process(float *dout, float *din, int n){
  // desc.: process elements of array 'din' & store the results in array 'dout'

  // returns a thread's index withinn a grid
  int
    i= threadIdx.x + blockIdx.x * blockDim.x,
    stride= blockDim.x * gridDim.x;

  // processing; each thread is used to process ceil(n / stride) elements of the array 'din'
  for(int idx= i; idx < n; idx+=stride)
    dout[idx]= 1.5 * din[idx];
}

// function definiiton
float errorCheck(float *hout, float *hin, int n){

  float maxError= 0.0f, err= 0.0f;
  for(int i= 0; i < n; ++i){
    err= fabs(hout[i] - hin[i]);
    maxError= max(err, maxError);
  }
  return maxError;
}

int main(int argc, char **argv){

  // pretty printing
  cout << "Benchmarking: kernel execution + synchronization overhead b/w the Host & Device\n";
  cout << "Array size: " << SIZE << "\n";
  cout << "Grid: " << BLOCKS << " blocks, " << THREADS << " threads/block\n";

  // init
  int devId= 0;
  if (argc > 1)
    devId= atoi(argv[1]);
  cudaSetDevice(devId);
  // display device's properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, devId);
  printf("Device : %s\n\n", prop.name);

  // allocate memory
  size_t bytes= SIZE * sizeof(float);
  float
    *hin= new float[SIZE],
    *hout, *din, * dout;
  cudaMallocHost(reinterpret_cast<void**>(&hout), bytes);
  cudaMalloc(reinterpret_cast<void**>(&din), bytes);
  cudaMalloc(reinterpret_cast<void**>(&dout), bytes);

  // create a stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // intialize array
  // 1. array 'hin'
  initializeArray<<<BLOCKS, THREADS>>>(din, 1.5f, SIZE);
  cudaMemcpy(hin, din, bytes, cudaMemcpyDeviceToHost);
  // 2. array 'din'
  initializeArray<<<BLOCKS, THREADS>>>(din, 1.0f, SIZE);

  // synchronize after each kernel launch
  auto start= chrono::steady_clock::now();
  for(int_fast16_t i= 0; i < STEPS; ++i)
    for(int_fast16_t j= 0; j < NSYNC; ++j){
      process<<<BLOCKS, THREADS, 0, stream>>>(dout, din, SIZE);
      cudaStreamSynchronize(stream);
    }
  auto end= chrono::steady_clock::now();
  chrono::duration<float, std::micro> time= end-start;
  cout << "[Per-kernel sync] Avg host time per kernel exec + sync: "
     << time.count() / (STEPS * NSYNC) << " usec\n";
  float baseline= time.count();

  // error check
  cudaMemcpy(hout, dout, bytes, cudaMemcpyDeviceToHost);
  cout << "Max. absolute error: " << errorCheck(hout, hin, SIZE) << "\n\n";

  // operations in a stream complete in the sequence, they were launched
  start= chrono::steady_clock::now();
  for(int_fast16_t i= 0; i < STEPS; ++i){
    for(int_fast16_t j= 0; j < NSYNC; ++j){
      process<<<BLOCKS, THREADS, 0, stream>>>(dout, din, SIZE);
    }
    cudaStreamSynchronize(stream);
  }
  end= chrono::steady_clock::now();
  time= end-start;
  cout << "[Batched sync] Avg host time per stream exec + sync ("
     << NSYNC << " kernel launches per sync): "
     << time.count() / (STEPS * NSYNC) << " usec\n";

  // error check
  cudaMemcpy(hout, dout, bytes, cudaMemcpyDeviceToHost);
  cout << "Max. absolute error: " << errorCheck(hout, hin, SIZE) << "\n\n";

  // create and launch a graph
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal );
  for(int_fast16_t j= 0; j < NSYNC; ++j)
    process<<<BLOCKS, THREADS, 0, stream>>>(dout, din, SIZE);
  cudaStreamEndCapture(stream, &graph);
  cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

  // warm up
  cudaGraphLaunch(graphExec, stream);
  cudaStreamSynchronize(stream);

  start= chrono::steady_clock::now();
  for(int_fast16_t i= 0; i < STEPS; ++i){
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);
  }
  end= chrono::steady_clock::now();
  time= end-start;
  cout << "[CUDA Graph] Avg host time per graph exec + sync ("
     << NSYNC << " kernels per graph launch): "
     << time.count() / (STEPS * NSYNC) << " usec\n";
  // error check
  cudaMemcpy(hout, dout, bytes, cudaMemcpyDeviceToHost);
  cout << "Max. absolute error: " << errorCheck(hout, hin, SIZE) << "\n\n";
  float graphTime= time.count();

  cout << "Graph speedup over per-kernel sync: " << baseline / graphTime << "x\n";

  // clean up
  cudaGraphExecDestroy(graphExec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaFree(din);
  cudaFree(dout);
  cudaFreeHost(hout);
  delete[] hin;

  return 0;
}
