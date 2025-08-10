/**
 * @file stream.cu
 * @brief This file tests the core::Stream class.
 *
 * This file shows a proof of concept of how the core::Stream class might be
 * used. However, the core::Stream class is currently rarely used in this
 * repository since creating too many streams slows down the calculation.
 *
 * The correct working of this file can be inspected with the visual profiler
 * from Nvidia NSight Systems.
 */

#include "../src/core.cuh"

/**
 * @brief This kernel implements active waiting on the GPU.
 */
__global__ void _wait(core::Kernel_Tensor<int> x) {
    core::index_type idx = blockDim.x * blockIdx.x + threadIdx.x;
    core::index_type stride = gridDim.x * blockDim.x;

    for (; idx < x.size(); idx += stride) x[idx] = idx;
}

/**
 * @brief This function launches the _wait kernel.
 */
void wait(core::Device_Tensor<int> &x, core::Stream &stream) {
    _wait<<<1, 1, 0, stream>>>(x);
}

/**
 * @brief Main function to test the core::Stream class.
 */
int main() {
    core::Stream synchronizing_stream;
    core::Stream stream_1;
    core::Stream stream_2;
    core::Stream stream_3;
    core::Stream stream_4;

    core::Device_Tensor<int> x({1'000'000});

    // Expectation: all streams run in parallel.
    wait(x, synchronizing_stream);
    wait(x, stream_1);
    wait(x, stream_2);
    wait(x, stream_3);
    wait(x, stream_4);
    cudaDeviceSynchronize();

    // Expectation: stream_2 and stream_3 run in parallel after stream_1 and
    // before stream_4.
    wait(x, stream_1);
    synchronizing_stream.wait(stream_1);
    stream_2.wait(synchronizing_stream);
    stream_3.wait(stream_2);
    wait(x, stream_2);
    wait(x, stream_3);
    synchronizing_stream.wait(stream_2);
    synchronizing_stream.wait(stream_3);
    stream_4.wait(synchronizing_stream);
    wait(x, stream_4);
    cudaDeviceSynchronize();

    // Expectation: stream_2 and stream_3 run in parallel after stream_1 and
    // before stream_4.
    wait(x, stream_1);
    synchronizing_stream.wait(cudaStream_t(stream_1));
    stream_2.wait(cudaStream_t(synchronizing_stream));
    stream_3.wait(cudaStream_t(stream_2));
    wait(x, stream_2);
    wait(x, stream_3);
    synchronizing_stream.wait(cudaStream_t(stream_2));
    synchronizing_stream.wait(cudaStream_t(stream_3));
    stream_4.wait(cudaStream_t(synchronizing_stream));
    wait(x, stream_4);
    cudaDeviceSynchronize();

    // Expectation: stream_2 and stream_3 run in parallel after stream_1 and
    // before stream_4.
    wait(x, stream_1);
    stream_1.make_wait(core::default_stream);
    stream_2.wait(core::default_stream);
    stream_3.wait(core::default_stream);
    wait(x, stream_2);
    wait(x, stream_3);
    stream_2.make_wait(core::default_stream);
    stream_3.make_wait(core::default_stream);
    stream_4.wait(core::default_stream);
    wait(x, stream_4);
    cudaDeviceSynchronize();
}
