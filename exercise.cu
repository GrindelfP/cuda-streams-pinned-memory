#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <chrono>

// GPU kernel function
__global__ void kernel(float* dA, float* dB, float* dC, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j;
    float ab, sum = 0.f;

    if (i < size) {
        ab = dA[i] * dB[i];
        for (j = 0; j < 100; j++) {
            sum += sinf(j + ab);
        }
        dC[i] = sum;
    }
}

// CPU computation function
void cpu_compute(float* hA, float* hB, float* hC, int size) {
    for (int i = 0; i < size; ++i) {
        float ab = hA[i] * hB[i];
        float sum = 0.f;
        for (int j = 0; j < 100; ++j) {
            sum += sinf(j + ab);
        }
        hC[i] = sum;
    }
}

int main() {
    const int nStreams = 4;
    const int nThreads = 512;
    const int totalSize = 512 * 50000;
    const int size = totalSize / nStreams;
    const size_t memSize = size * sizeof(float);

    float* hA, * hB, * hC;
    float* dA, * dB, * dC;

    // Allocate pinned memory on host
    cudaMallocHost(&hA, totalSize * sizeof(float));
    cudaMallocHost(&hB, totalSize * sizeof(float));
    cudaMallocHost(&hC, totalSize * sizeof(float));

    // Allocate device memory
    cudaMalloc(&dA, totalSize * sizeof(float));
    cudaMalloc(&dB, totalSize * sizeof(float));
    cudaMalloc(&dC, totalSize * sizeof(float));

    // Initialize host arrays
    for (int i = 0; i < totalSize; ++i) {
        hA[i] = sinf(i);
        hB[i] = cosf(2.0f * i - 5.0f);
        hC[i] = 0.0f;
    }

    // Create CUDA streams
    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    int nBlocks = (size + nThreads - 1) / nThreads;

    // GPU computation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < nStreams; ++i) {
        cudaMemcpyAsync(dA + i * size, hA + i * size, memSize, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(dB + i * size, hB + i * size, memSize, cudaMemcpyHostToDevice, streams[i]);
    }

    for (int i = 0; i < nStreams; ++i) {
        kernel << <nBlocks, nThreads, 0, streams[i] >> > (dA + i * size, dB + i * size, dC + i * size, size);
    }

    for (int i = 0; i < nStreams; ++i) {
        cudaMemcpyAsync(hC + i * size, dC + i * size, memSize, cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);

    // CPU computation
    auto cpuStart = std::chrono::high_resolution_clock::now();
    cpu_compute(hA, hB, hC, totalSize);
    auto cpuStop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> cpuDuration = cpuStop - cpuStart;

    // Output results
    std::cout << "GPU calculation time: " << gpuTime << " ms\n";
    std::cout << "CPU calculation time: " << cpuDuration.count() << " ms\n";
    std::cout << "Rate: " << cpuDuration.count() / gpuTime << "x\n";

    // Clean up
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
