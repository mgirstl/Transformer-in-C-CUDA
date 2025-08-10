/**
 * @file curand.cuh
 * @brief This file contains wrappers around useful cuRAND functions for easier
 * usage.
 *
 * This file is part of the `core` namespace, which provides fundamental types,
 * utilities, and functions that are widely used across different modules of
 * the project.
 */

#pragma once

#include <curand_kernel.h>

#include "cuda.cuh"
#include "device_tensor.cuh"
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
 * This specific file defines wrappers for some cuRAND functions.
 *
 * @endinternal
 */
namespace core {

    /**
     * @brief Draw samples from a uniform distribution.
     *
     * This function draws and saves uniform random values from the interval
     * (0, 1] into a tensor using the cuRAND host API.
     *
     * @param generator The cuRAND generator used to produce random numbers.
     * @param tensor The tensor where the generated random values will be
     * stored in.
     * @return The status of the cuRAND operation.
     */
    template <float_type dtype>
    curandStatus_t inline generate_uniform(curandGenerator_t &generator,
                                           Device_Tensor<dtype> &tensor) {
        curandStatus_t error;

        if constexpr (std::is_same_v<dtype, float>)
            error = checkCuda(
                curandGenerateUniform(generator, tensor.data(), tensor.size()));

        else if constexpr (std::is_same_v<dtype, double>)
            error = checkCuda(curandGenerateUniformDouble(
                generator, tensor.data(), tensor.size()));

        else
            static_assert("Unsupported dtype for core::generate_uniform!");

        checkCuda(cudaPeekAtLastError());
        return error;
    }

    /**
     * @brief Draw samples from a uniform distribution.
     *
     * This function draws and saves uniform random values from the interval
     * (0, 1] into a tensor using the cuRAND device API.
     *
     * @note The types float4 and double2 are only supported with
     * curandStatePhilox4_32_10_t.
     *
     * @tparam dtype The data type of the generated random numbers. Supported
     * types are float, double, float4, and double2.
     * @tparam curand_state Any valid cuRAND state type.
     * @param state The current state of the generator.
     * @return The generated random number(s).
     */
    template <typename dtype, typename curand_state>
    __device__ dtype inline uniform(curand_state &state) {
        if constexpr (std::is_same_v<dtype, float>)
            return curand_uniform(&state);

        else if constexpr (std::is_same_v<dtype, double>)
            return curand_uniform_double(&state);

        else if constexpr (std::is_same_v<dtype, float4> &&
                           std::is_same_v<curand_state,
                                          curandStatePhilox4_32_10_t>)
            return curand_uniform4(&state);

        else if constexpr (std::is_same_v<dtype, double2> &&
                           std::is_same_v<curand_state,
                                          curandStatePhilox4_32_10_t>)
            return curand_uniform2_double(&state);

        else
            static_assert("Unsupported dtype for core::uniform!");
    }

    /**
     * @brief Draw samples from a normal distribution with a given mean and
     * standard deviation.
     *
     * This function draws normally distributed random values and saves them
     * into a tensor using the cuRAND host API.
     *
     * @param generator The cuRAND generator used to produce random numbers.
     * @param tensor The tensor where the generated random values will be
     * stored in.
     * @param mean The mean of the normal distribution.
     * @param std The standard deviation of the normal distribution.
     * @return The status of the cuRAND operation.
     */
    template <float_type dtype>
    curandStatus_t inline generate_normal(curandGenerator_t &generator,
                                          Device_Tensor<dtype> &tensor,
                                          const dtype mean = 0,
                                          const dtype std = 1) {
        curandStatus_t error;

        if constexpr (std::is_same_v<dtype, float>)
            error = checkCuda(curandGenerateNormal(generator, tensor.data(),
                                                   tensor.size(), mean, std));

        else if constexpr (std::is_same_v<dtype, double>)
            error = checkCuda(curandGenerateNormalDouble(
                generator, tensor.data(), tensor.size(), mean, std));

        else
            static_assert("Unsupported dtype for core::generate_normal!");

        checkCuda(cudaPeekAtLastError());
        return error;
    }

}  // namespace core
