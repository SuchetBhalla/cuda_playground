# Learning CUDA

reference: [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)

## Device Information

<pre>+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   50C    P8             10W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0 </pre>

## 1_data_transfer.cu

### Results

Transfer size (MB): 16

Pageable transfers
-	Bandwidth (GB/sec), host to device: 3.96
-	Bandwidth (GB/sec), device to host: 1.58

Pinned transfers
-	Bandwidth (GB/sec), host to device: 12.18
-	Bandwidth (GB/sec), device to host: 13.01

### Conclusion

Data transfer between the host & device(s) is faster via Pinned memory (on the host).

**Tradeoff:** Pinned memory becomes unavailable to the host; thus limiting the amount of data which can be stored
in the host's memory by its processes (including the OS).

reference: [How to Optimize Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)

## 2_matrix_transpose.cu

### Results

Matrix size: 1024 x 1024

Tile size: 32 x 32 x 1

Block size: 32 x 8 x 1

| Routine | Bandwidth (GB/sec) |
| --- | --- |
| copy | 212.45|
| copysram | 227.54|
| transposeNaive | 46.68|
| transposeSubOptima | 78.59|
| transposeOptima | 204.18|

### Conclusion
The optimal way to read from & write to global memory, is to access contiguous addresses.
If this can be achieved for all threads in a warp, then 32 accesses are coalesced into one.

### HowTo

1. All addresses in shared memory, are accessed in 1 cycle, by the threads in a warp, even if the accesses are random;
provided bank conflicts do not occur.
2. Bank: The shared memory is split into 32 banks (~one bank per thread in a warp). A bank can supply 32 bits in 1 cycle, to a thread.
- A bank conflict is avoided when the 32 threads access different banks.
- Bank index = (address / 4) % 32
- source: ChatGPT

## 3_async.cu

### Results

**Device:** Tesla T4

Concurrency is possible.
- Engine count: 3

Time for synchronous data-transfer & execution: 3.027 msec

Time for asynchronous (v1) data-transfer & execution: 1.991 msec

Time for asynchronous (v2) data-transfer & execution: 1.824 msec


### Information

Operations on the device can be performed concurrently aka *overlapped*, e.g., data transfer (between the host & device) and a computation.

CUDA Streams: There is 1 default stream, and I can create more streams using the CUDA API.

Difference:
1. Default stream: Memcpy operations issued to the default stream, between the host and device are synchronizing aka blocking, i.e., the CPU waits.
2. Non-default stream: the operations are non-blocking, thus the CPU is available to run processes.

Synchronizing non-default streams: APIs exist to synchronize
1. a stream to a specific event in another stream (even if on a different device)
2. the host to a specific event (in a stream)
3. the host to a specific stream; when the issued operations complete.
- This can be either blocking or not.
4. the host to all streams; when the issued operations complete


Conditions which enable concurrency:
1. the device has separate engines for: kernel execution & data transfer
- This can be queried from ~~either the field *deviceOverlap* of a struct *cudaDeviceProp*,~~
2. the operations (to be overlapped) belong to different, non-default streams
3. the host memory involved is pinned memory

**reference**: [How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)
