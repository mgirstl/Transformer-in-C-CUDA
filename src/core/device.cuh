/**
 * @file device.cuh
 * @brief Implements the Device class, which provides additional information
 * such as the recommended number of blocks and threads.
 *
 * This file is part of the `core` namespace, which provides fundamental types,
 * utilities, and functions that are widely used across different modules of
 * the project.
 */

#pragma once

#include <cublas_v2.h>
#include <curand.h>

#include <cmath>
#include <limits>

#include "cuda.cuh"
#include "stream.cuh"
#include "types.cuh"

/**
 * @internal
 *
 * @namespace core
 * @brief Provides core facilities used throughout the entire codebase.
 *
 * The `core` namespace contains fundamental types, utilities, and functions
 * that are widely used across different modules of the project.
 *
 * This specific file defines the Device class.
 *
 * @endinternal
 */
namespace core {

    /**
     * @class Device
     * @brief Calculates the recommended number of threads and blocks.
     * Also manages the current cublasHandle and curandGenerator.
     *
     * @attention It is recommended to pass this class by reference to reduce
     * cuBLAS and cuRAND setup times.
     */
    class Device {
      private:
        cublasHandle_t _cublas_handle;
        curandGenerator_t _curand_generator;

        // Maximum number of threads supported by the device.
        index_type _max_threads;

        /**
         * @brief Number of threads chosen based on testing for satisfactory
         * performance.
         */
        index_type _threads_1D = 256;
        index_type _threads_2D = 16;
        index_type _threads_3D = 8;

        /**
         * @brief Upper bound for the number of blocks chosen based on testing
         * for satisfactory performance.
         */
        index_type _max_blocks;
        index_type _blocks_1D;
        index_type _blocks_2D;
        index_type _blocks_3D;

      public:
        /**
         * @brief Construct a new Device object.
         *
         * @param curand_rng_type The type used for generating the
         * curandGenerator. It specifies which algorithm is used for generating
         * the random numbers.
         *
         * More information:
         * https://docs.nvidia.com/cuda/curand/host-api-overview.html
         */
        Device(const curandRngType_t curand_rng_type =
                   CURAND_RNG_PSEUDO_PHILOX4_32_10) {
            int id;
            checkCuda(cudaGetDevice(&id));

            cudaDeviceProp properties;
            checkCuda(cudaGetDeviceProperties(&properties, id));

            _max_threads = properties.maxThreadsPerBlock;
            _max_blocks = properties.multiProcessorCount * 32;

            _blocks_1D = _max_blocks;
            _blocks_2D = sqrt(_blocks_1D);
            _blocks_3D = pow(_blocks_1D, 1. / 3.);

            checkCuda(curandCreateGenerator(&_curand_generator,
                                            CURAND_RNG_PSEUDO_PHILOX4_32_10));
            checkCuda(curandSetPseudoRandomGeneratorSeed(_curand_generator, 0));
            checkCuda(cublasCreate(&_cublas_handle));
        }

        ~Device() {
            checkCuda(curandDestroyGenerator(_curand_generator));
            checkCuda(cublasDestroy(_cublas_handle));
        }

        /**
         * @brief Returns the maximum number of threads the device supports.
         */
        index_type max_threads() const { return _max_threads; }

        /**
         * @brief Returns the recommended number of threads for one-dimensional
         * grids.
         */
        index_type threads() const { return _threads_1D; }

        /**
         * @brief Returns the recommended number of threads for one-dimensional
         * grids.
         */
        index_type threads_1D() const { return _threads_1D; }

        /**
         * @brief Returns the recommended number of threads for two-dimensional
         * grids.
         */
        dim3 threads_2D() const { return dim3(_threads_2D, _threads_2D, 1); }

        /**
         * @brief Returns the recommended number of threads for
         * three-dimensional grids.
         */
        dim3 threads_3D() const {
            return dim3(_threads_3D, _threads_3D, _threads_3D);
        }

        /**
         * @brief Returns the maximum recommended number of blocks.
         */
        index_type max_blocks() const { return _max_blocks; }

        /**
         * @brief Returns the maximum recommended number of blocks for
         * one-dimensional grids.
         */
        index_type blocks() const { return _blocks_1D; }

        /**
         * @brief Returns the maximum recommended number of blocks for
         * one-dimensional grids.
         */
        index_type blocks_1D() const { return _blocks_1D; }

        /**
         * @brief Returns the maximum recommended number of blocks for
         * two-dimensional grids.
         */
        dim3 blocks_2D() const { return dim3(_blocks_2D, _blocks_2D, 1); }

        /**
         * @brief Returns the maximum recommended number of blocks for
         * three-dimensional grids.
         */
        dim3 blocks_3D() const {
            return dim3(_blocks_3D, _blocks_3D, _blocks_3D);
        }

        /**
         * @brief Returns the recommended number of blocks for one-dimensional
         * grids.
         *
         * @note Assumes that the kernel uses grid-strided-loops.
         *
         * @param elements The number of elements.
         * @return The recommended number of blocks.
         */
        index_type blocks(const index_type elements) const {
            return calc_blocks(elements, _threads_1D, _blocks_1D);
        }

        /**
         * @brief Returns the recommended number of blocks for two-dimensional
         * grids.
         *
         * @note Assumes that the kernel uses grid-strided-loops.
         *
         * @param elements_x The number of elements in the x-direction.
         * @param elements_y The number of elements in the y-direction.
         * @return The recommended number of blocks.
         */
        dim3 blocks_2D(const index_type elements_x,
                       const index_type elements_y) const {
            index_type _blocks_x =
                calc_blocks(elements_x, _threads_2D, _blocks_2D);
            index_type _blocks_y =
                calc_blocks(elements_y, _threads_2D, _blocks_2D);
            index_type _blocks_z = 1;
            return dim3(_blocks_x, _blocks_y, _blocks_z);
        }

        /**
         * @brief Returns the recommended number of blocks for three-dimensional
         * grids.
         *
         * @note Assumes that the kernel uses grid-strided-loops.
         *
         * @param elements_x The number of elements in the x-direction.
         * @param elements_y The number of elements in the y-direction.
         * @param elements_z The number of elements in the z-direction.
         * @return The recommended number of blocks.
         */
        dim3 blocks_3D(const index_type elements_x, const index_type elements_y,
                       const index_type elements_z) const {
            index_type _blocks_x =
                calc_blocks(elements_x, _threads_3D, _blocks_3D);
            index_type _blocks_y =
                calc_blocks(elements_y, _threads_3D, _blocks_3D);
            index_type _blocks_z =
                calc_blocks(elements_z, _threads_3D, _blocks_3D);
            return dim3(_blocks_x, _blocks_y, _blocks_z);
        }

        /**
         * @brief Returns the cuBLAS handle.
         *
         * @param stream The stream used for the cuBLAS operation.
         * @return The cuBLAS handle.
         */
        cublasHandle_t &cublas_handle(
            const cudaStream_t stream = default_stream) {
            checkCuda(cublasSetStream(_cublas_handle, stream));
            return _cublas_handle;
        }

        /**
         * @brief Returns the cuRAND generator.
         *
         * @param stream The stream used for the cuRAND operation.
         * @return The cuRAND generator.
         */
        curandGenerator_t &curand_generator(
            const cudaStream_t stream = default_stream) {
            checkCuda(curandSetStream(_curand_generator, stream));
            return _curand_generator;
        }

      private:
        /**
         * @brief Calculates the number of blocks needed depending on the number
         * of elements that need to be processed and the number of threads.
         *
         * @param elements The number of elements to be processed.
         * @param threads The number of threads used.
         * @param blocks The maximum number of blocks. This is recommended to
         * be set if grid-strided-loops are used.
         * @return The recommended number of blocks.
         */
        static index_type calc_blocks(
            const index_type &elements, const index_type &threads,
            const index_type &blocks = std::numeric_limits<index_type>::max()) {
            index_type _blocks = (elements + threads - 1) / threads;
            return _blocks < blocks ? _blocks : blocks;
        }
    };

}  // namespace core
