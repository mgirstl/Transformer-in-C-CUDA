/**
 * @file random_number_generator.cuh
 * @brief This file implements the Random_Number_Generator singleton.
 *
 * This file is part of the `utils` namespace, which implements utility
 * functions and classes.
 */

#pragma once

#include <curand_kernel.h>

#include "../core.cuh"

/**
 * @internal
 *
 * @namespace utils
 * @brief Namespace for utility functions and classes.
 *
 * This specific file defines the Random_Number_Generator singleton.
 *
 * @endinternal
 */
namespace utils {

    /**
     * @brief This kernel initializes the curand states.
     *
     * @attention This template specification is for cases when the
     * `curand_state` is not `curandStatePhilox4_32_10_t`. In testing, setting a
     * different sequence number with the same seed for each state led to the
     * same random numbers in most threads. Setting different seeds for each
     * state solves this issue.
     * However, this can lead to issues with the statistical properties of the
     * random numbers. Hence, use this version with caution. It is the default
     * version because it should work with any valid cuRAND state. But if
     * possible, a template specification as used for
     * `curandStatePhilox4_32_10_t` is preferred.
     * This means, users should verify if the problem persists for other cuRAND
     * states. If the issue does not occur, they should implement a template
     * specification for those states as well.
     *
     * More information:
     * https://docs.nvidia.com/cuda/curand/device-api-overview.html
     *
     * @param state Pointer to the array of curand states. The array must be
     * at least as large as the maximum number of unique threads that will be
     * launched.
     */
    template <typename curand_state>
    __global__ void setup_rng(curand_state *state) {
        core::index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(idx, 0, 0, &state[idx]);
    }

    /**
     * @brief This kernel initializes the curand states.
     *
     * @param state Pointer to the array of curand states. The array must be
     * at least as large as the maximum number of unique threads that will be
     * launched.
     */
    template <>
    __global__ void setup_rng<curandStatePhilox4_32_10_t>(
        curandStatePhilox4_32_10_t *state) {
        core::index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(0, idx, 0, &state[idx]);
    }

    /**
     * @class Random_Number_Generator
     * @brief A class to hold the `curandStates` used by the cuRAND device API.
     *
     * This class is used to provide the entire codebase with the same
     * `curandStates`, minimizing state setup times and memory usage.
     *
     * @warning Ensure that kernels using these states are not launched
     * concurrently in different streams.
     *
     * @attention The number of states created depends on the `core::Device`
     * provided during the initialization of this singleton. Therefore, the same
     * Device must be used wherever this singleton is utilized. Additionally,
     * one should not use more threads and blocks than recommended by the
     * `core::Device`.
     *
     * @tparam curand_state needs to be a valid cuRAND state type, e.g.,
     * `curandState_t`, `curandStateXORWOW_t`, or `curandStatePhilox4_32_10_t`.
     * This type specifies which algorithm is used for generating the random
     * numbers.
     *
     * More information:
     * https://docs.nvidia.com/cuda/curand/device-api-overview.html
     */
    template <typename curand_state>
    class Random_Number_Generator {
      private:
        curand_state *_states;
        core::index_type _size;
        static core::index_type instance_counter;

        /**
         * @brief Constructs a new Random_Number_Generator object.
         *
         * @param device The device object that calculates the maximum number of
         * threads and blocks.
         * @param stream The CUDA stream used for launching the `setup_rng`
         * kernel.
         */
        Random_Number_Generator(
            core::Device &device,
            const cudaStream_t stream = core::default_stream) {
            core::index_type blocks = device.max_blocks();
            core::index_type threads = device.max_threads();

            _size = blocks * threads * sizeof(curand_state);
            core::checkCuda(cudaMalloc(&_states, _size));
            setup_rng<<<blocks, threads, 0, stream>>>(_states);
            core::checkCuda(cudaPeekAtLastError());
        }

        /**
         * @brief Destructor to free the curand states.
         */
        ~Random_Number_Generator() { core::checkCuda(cudaFree(_states)); }

      public:
        /**
         * @brief Get the singleton instance of the Random_Number_Generator.
         *
         * @param device The device object used for initialization.
         * @param stream The CUDA stream  used for initialization.
         * @return The singleton instance of the Random_Number_Generator.
         */
        static Random_Number_Generator &get_instance(
            core::Device &device,
            const cudaStream_t stream = core::default_stream) {
            static Random_Number_Generator instance(device, stream);
            ++instance_counter;
            return instance;
        }

        /**
         * @brief Get the pointer to the curand states array.
         */
        curand_state *states() { return _states; }

        // Delete copy constructor and assignment operator to enforce singleton
        // pattern.
        Random_Number_Generator(const Random_Number_Generator &) = delete;
        Random_Number_Generator &operator=(const Random_Number_Generator &) =
            delete;

        /**
         * @brief Returns the memory footprint of the object on GPU.
         *
         * @param average If true, returns the average size per instance. If
         * false, the total size is returned.
         * @return The memory footprint.
         */
        const core::Mem_Size mem_size(const bool average = false) {
            size_t fixed_size = 0;
            if (average)
                fixed_size += (_size + instance_counter - 1) / instance_counter;
            else
                fixed_size += _size;

            size_t variable_size = 0;
            return {fixed_size, variable_size};
        }
    };

    template <typename curand_state>
    core::index_type Random_Number_Generator<curand_state>::instance_counter =
        0;

}  // namespace utils
