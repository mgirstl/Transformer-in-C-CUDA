/**
 * @file atomic.cu
 * @brief This file tests if cuda atomics are working. It does not rely on any
 * other files in this framework.
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

constexpr int threads = 2;
constexpr int blocks = 2;
constexpr int size = threads * blocks;

/**
 * @brief Prints the cuda error string if an error has occurred.
 *
 * @param error The error which occurred.
 * @return The original error.
 */
#define checkCuda(val) checkCudaErrors((val), #val, __FILE__, __LINE__)

/**
 * @brief Prints the CUDA error string if an error has occurred.
 *
 * @note Usually this function is not called directly. Instead, use the
 * macro checkCuda, which provides all additional arguments.
 *
 * @param error The error which occurred.
 * @param func The function name where the error occurred.
 * @param file The file name where the error occurred.
 * @param line The line where the error occurred.
 * @return The original error.
 */
inline cudaError_t checkCudaErrors(cudaError_t error, char const *const func,
                                   const char *const file, int const line) {
    if (error != cudaSuccess) {
        std::cout << "CUDA Runtime Error: " << cudaGetErrorString(error)
                  << "\n\tat " << file << ":" << line << " '" << func << "'\n";
    }
    return error;
}

/**
 * @brief This kernel adds 1 to each element in the data array using atomicAdd.
 */
__global__ void add(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        printf("device_data at index %d: %f (before add)\n", idx, data[idx]);
        atomicAdd(&data[idx], 1.0);
        printf("device_data at index %d: %f (after add)\n", idx, data[idx]);
    }
}

/**
 * @brief Main function to test CUDA atomic operations.
 */
int main() {
    float *device_data, *host_data;

    checkCuda(cudaMalloc(&device_data, size * sizeof(float)));
    checkCuda(cudaMallocHost(&host_data, size * sizeof(float)));

    for (int i = 0; i < size; ++i) {
        host_data[i] = 1.0;
        std::cout << "host_data at index " << i << ": " << host_data[i]
                  << std::endl;
    }

    checkCuda(cudaMemcpy(device_data, host_data, size * sizeof(float),
                         cudaMemcpyHostToDevice));

    add<<<blocks, threads>>>(device_data);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(host_data, device_data, size * sizeof(float),
                         cudaMemcpyDeviceToHost));

    for (int i = 0; i < size; ++i)
        if (host_data[i] < 1.5)
            std::cout << "Error at index " << i << ": " << host_data[i]
                      << std::endl;

    checkCuda(cudaFree(device_data));
    checkCuda(cudaFreeHost(host_data));
}
