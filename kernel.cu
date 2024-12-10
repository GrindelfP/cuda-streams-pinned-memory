#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <chrono>

/**
* Kernel function for the program.
*/
__global__
void
kernel(
    float* dA,
    float* dB,
    float* dC,
    int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j;
    float ab, sum = 0.f;

    if (i < size) {
        ab = dA[i] * dB[i];

        for (j = 0; j < 1000; ++j) sum += sinf(j + ab);

        dC[i] = sum;
    }
}

/**
* CPU function for the program.
*/
void
cpu_compute(
    float* hA,
    float* hB,
    float* hC,
    int size
) {
    for (int i = 0; i < size; ++i) {
        float ab = hA[i] * hB[i];
        float sum = 0.f;

        for (int j = 0; j < 1000; ++j)  sum += sinf(j + ab);

        hC[i] = sum;
    }
}

void
testGPU(
    const int nStreams,
    float* hA,
    float *hB,
    float *hC_GPU, 
    std::chrono::duration<float, std::milli> cpuDuration
) {
    // ===================
    // DATA INITIALIZATION
    // ===================
    const int nThreads = 512, totalSize = 512 * 50000, size = totalSize / nStreams;
    const size_t memSize = size * sizeof(float), arraySize = totalSize * sizeof(float);

    float* dA, * dB, * dC;

    cudaMalloc((void**)&dA, arraySize);
    cudaMalloc((void**)&dB, arraySize);
    cudaMalloc((void**)&dC, arraySize);

    // ================================== 
    // GPU computation and data maagement
    // ================================== 
    cudaStream_t *streams = new cudaStream_t[nStreams];

    for (int i = 0; i < nStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    int nBlocks = (size + nThreads - 1) / nThreads;

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
        cudaMemcpyAsync(hC_GPU + i * size, dC + i * size, memSize, cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);
    std::cout << "GPU operation done with number of streams: " << nStreams << ".\n";

    std::cout << "GPU calculation time: " << gpuTime << " ms\n";
    std::cout << "Rate: " << cpuDuration.count() / gpuTime << "x\n";

    // ========
    // CLEAN UP
    // ========
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] streams;
}

/**
* Main function for the program.
*/
int
main() {
    // ===================
    // DATA INITIALIZATION
    // ===================
    const int streamsNumbers[] = { 1, 2, 4 };
    const int nThreads = 512, totalSize = 512 * 50000;
    const size_t arraySize = totalSize * sizeof(float);

    float* hA, * hB, * hC, * hC_GPU;

    cudaMallocHost((void**)&hA, arraySize);
    cudaMallocHost((void**)&hB, arraySize);
    cudaMallocHost((void**)&hC, arraySize);
    cudaMallocHost((void**)&hC_GPU, arraySize);

    for (int i = 0; i < totalSize; ++i) {
        hA[i] = sinf(i);
        hB[i] = cosf(2.0f * i - 5.0f);
        hC[i] = 0.0f;
        hC_GPU[i] = 0.0f;
    }

    // ===============
    // CPU COMPUTATION
    // ===============
    auto cpuStart = std::chrono::high_resolution_clock::now();
    cpu_compute(hA, hB, hC, totalSize);
    auto cpuStop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> cpuDuration = cpuStop - cpuStart;
    std::cout << "CPU operation done!\n";
    std::cout << "CPU calculation time: " << cpuDuration.count() << " ms\n";
    std::cout << "Number of streams: 1\n\n";

    // ===============
    // GPU COMPUTATION
    // ===============
    for (int i = 0; i < 3; ++i) testGPU(streamsNumbers[i], hA, hB, hC_GPU, cpuDuration);

    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);
    cudaFreeHost(hC_GPU);

    return 0;
}
