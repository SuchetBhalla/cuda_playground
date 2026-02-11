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

Transfer size (MB): 16

Pageable transfers
-	Bandwidth (GB/sec), host to device: 3.96
-	Bandwidth (GB/sec), device to host: 1.58

Pinned transfers
-	Bandwidth (GB/sec), host to device: 12.18
-	Bandwidth (GB/sec), device to host: 13.01

## 2_matrix_transpose.cu

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
