/**
 * @file compare.cuh
 * @brief Implements functions to compare different tensors, launch testing
 * kernels and measure statistical properties of tensors and compare them with
 * an expected value.
 *
 * This file is part of the `test` namespace, which implements classes and
 * functions for testing.
 */

#pragma once

#include <cmath>
#include <iostream>
#include <limits>

#include "../core.cuh"

/**
 * @internal
 *
 * @namespace test
 * @brief This namespace contains functions and classes for testing.
 *
 * This specific file defines functions to compare different tensors, launch
 * testing kernels and measure statistical properties of tensors and compare
 * them with an expected value.
 *
 * @endinternal
 */
namespace test {

    /**
     * @brief Calculates the absolute value of a variable for all arithmetic
     * types.
     *
     * @note This function is needed because `std::abs` is not overloaded for
     * unsigned types.
     */
    template <core::arithmetic_type data_type>
    __host__ __device__ data_type inline abs(const data_type &x) {
        if (x >= 0)
            return x;
        else
            return -x;
    }

    /**
     * @brief Compares two values to determine if they are equal using epsilon
     * comparison.
     *
     * @param x The first value.
     * @param y The second value.
     * @param scale Scales the precision of the epsilon comparison.
     * @return false If both values are `NAN` or have a larger difference than
     * `scale * max(abs(x), abs(y), 1) *
     * std::numeric_limits<data_type>::epsilon()`.
     * @return true Otherwise.
     */
    template <core::arithmetic_type data_type>
    __host__ __device__ bool inline compare(const data_type &x,
                                            const data_type &y,
                                            const data_type scale = 100) {
        if (std::isnan(x) || std::isnan(y)) return false;

        if (x == y) return true;

        // Absolute epsilon
        data_type epsilon = scale * std::numeric_limits<data_type>::epsilon();

        // Relative epsilon
        data_type max = 1;
        if (abs(x) > max) max = abs(x);
        if (abs(y) > max) max = abs(y);
        epsilon *= max;

        // Comparison which also works for unsigned values.
        if (x > y && x - y > epsilon) return false;
        if (y > x && y - x > epsilon) return false;

        return true;
    }

    /**
     * @brief Compares two Host_Tensor via epsilon comparison.
     *
     * For details on the epsilon comparison:
     * @see test::compare
     *
     * @param x The first tensor.
     * @param y The second tensor.
     * @param scale Scales the precision of the epsilon comparison.
     * @return true If x and y are equal.
     * @return false If x and y are not equal.
     */
    template <core::arithmetic_type data_type>
    bool compare(const core::Host_Tensor<data_type> &x,
                 const core::Host_Tensor<data_type> &y,
                 const data_type scale = 100) {
        if (x.rank() != y.rank()) return false;
        if (x.size() != y.size()) return false;
        if (x.batchsize() != y.batchsize()) return false;

        for (core::index_type idx = 0; idx < x.rank(); ++idx)
            if (x.shape(idx) != y.shape(idx)) return false;

        for (core::index_type idx = 0; idx < x.size(); ++idx)
            if (!compare(x[idx], y[idx], scale)) return false;

        return true;
    }

    /**
     * @brief Compares a Device_Tensor with a Host_Tensor via epsilon
     * comparison.
     *
     * For details on the epsilon comparison:
     * @see test::compare
     *
     * @param x The first tensor.
     * @param y The second tensor.
     * @param scale Scales the precision of the epsilon comparison.
     * @return true If x and y are equal.
     * @return false If x and y are not equal.
     */
    template <core::arithmetic_type data_type>
    bool compare(const core::Device_Tensor<data_type> &x,
                 const core::Host_Tensor<data_type> &y,
                 const data_type scale = 100) {
        auto _x = core::Host_Tensor(x);
        return compare(_x, y, scale);
    }

    /**
     * @brief Compares a Host_Tensor with a Device_Tensor via epsilon
     * comparison.
     *
     * For details on the epsilon comparison:
     * @see test::compare
     *
     * @param x The first tensor.
     * @param y The second tensor.
     * @param scale Scales the precision of the epsilon comparison.
     * @return true If x and y are equal.
     * @return false If x and y are not equal.
     */
    template <core::arithmetic_type data_type>
    bool compare(const core::Host_Tensor<data_type> &x,
                 const core::Device_Tensor<data_type> &y,
                 const data_type scale = 100) {
        auto _y = core::Host_Tensor(y);
        return compare(x, _y, scale);
    }

    /**
     * @brief Compares two Device_Tensor via epsilon comparison.
     *
     * For details on the epsilon comparison:
     * @see test::compare
     *
     * @param x The first tensor.
     * @param y The second tensor.
     * @param scale Scales the precision of the epsilon comparison.
     * @return true If x and y are equal.
     * @return false If x and y are not equal.
     */
    template <core::arithmetic_type data_type>
    bool compare(const core::Device_Tensor<data_type> &x,
                 const core::Device_Tensor<data_type> &y,
                 const data_type scale = 100) {
        auto _x = core::Host_Tensor(x);
        auto _y = core::Host_Tensor(y);
        return compare(_x, _y, scale);
    }

    /**
     * @brief Compares the mean of a tensor to an expected mean value within a
     * 99.7% confidence interval.
     *
     * @param tensor The tensor whose mean is to be compared.
     * @param mean_expected The expected mean value.
     * @param variance_expected The expected variance value.
     * @param print If true, then additional debug information is printed to
     * `std::cout`.
     * @return true If the mean of the tensor is within the confidence interval
     * of the expected mean.
     * @return false Otherwise.
     */
    template <core::arithmetic_type data_type>
    bool compare_mean(const core::Host_Tensor<data_type> &tensor,
                      const data_type &mean_expected,
                      const data_type &variance_expected,
                      const bool print = false) {
        data_type mean = 0;
        for (core::index_type idx = 0; idx < tensor.size(); ++idx)
            mean += tensor[idx];
        mean /= tensor.size();

        data_type error = sqrt(variance_expected / tensor.size());
        error *= 3;  // 99.7% confidence

        if (print)
            std::cout << "Calculated Mean: " << mean
                      << ", Expected Mean: " << mean_expected
                      << ", Error: " << error << std::endl;

        return abs(mean - mean_expected) < error;
    }

    /**
     * @brief Compares the mean of a tensor to an expected mean value within a
     * 99.7% confidence interval.
     *
     * @param tensor The tensor whose mean is to be compared.
     * @param mean_expected The expected mean value.
     * @param variance_expected The expected variance value.
     * @param print If true, then additional debug information is printed to
     * `std::cout`.
     * @return true If the mean of the tensor is within the confidence interval
     * of the expected mean.
     * @return false Otherwise.
     */
    template <core::arithmetic_type data_type>
    bool compare_mean(const core::Device_Tensor<data_type> &tensor,
                      const data_type &mean_expected,
                      const data_type &variance_expected,
                      const bool print = false) {
        auto _tensor = core::Host_Tensor(tensor);
        return compare_mean(_tensor, mean_expected, variance_expected, print);
    }

    /**
     * @brief Compares the variance of a tensor to an expected variance value
     * within a 99.7% confidence interval.
     *
     * @param tensor The tensor whose mean is to be compared.
     * @param mean_expected The expected mean value.
     * @param variance_expected The expected variance value.
     * @param print If true, then additional debug information is printed to
     * `std::cout`.
     * @return true If the variance of the tensor is within the confidence
     * interval of the expected variance.
     * @return false Otherwise.
     */
    template <core::arithmetic_type data_type>
    bool compare_variance(const core::Host_Tensor<data_type> &tensor,
                          const data_type &mean_expected,
                          const data_type &variance_expected,
                          const bool print = false) {
        data_type variance = 0;
        for (core::index_type idx = 0; idx < tensor.size(); ++idx)
            variance +=
                (tensor[idx] - mean_expected) * (tensor[idx] - mean_expected);
        variance /= tensor.size() - 1;

        data_type error = variance_expected * sqrt(2. / (tensor.size() - 1));
        error *= 3;  // 99.7% confidence

        if (print)
            std::cout << "Calculated Variance: " << variance
                      << ", Expected Variance: " << variance_expected
                      << ", Error: " << error << std::endl;

        return abs(variance - variance_expected) < error;
    }

    /**
     * @brief Compares the variance of a tensor to an expected variance value
     * within a 99.7% confidence interval.
     *
     * @param tensor The tensor whose mean is to be compared.
     * @param mean_expected The expected mean value.
     * @param variance_expected The expected variance value.
     * @param print If true, then additional debug information is printed to
     * `std::cout`.
     * @return true If the variance of the tensor is within the confidence
     * interval of the expected variance.
     * @return false Otherwise.
     */
    template <core::arithmetic_type data_type>
    bool compare_variance(const core::Device_Tensor<data_type> &tensor,
                          const data_type &mean_expected,
                          const data_type &variance_expected,
                          const bool print = false) {
        auto _tensor = core::Host_Tensor(tensor);
        return compare_variance(_tensor, mean_expected, variance_expected,
                                print);
    }

    /**
     * @brief A CUDA kernel that executes a given device function.
     *
     * For more information see:
     * @see test::test_kernel
     *
     * @tparam function A function that can be invoked with a `bool * const`
     * argument.
     * @param kernel_function The `__device__` function to be invoked inside the
     * kernel. It must take a `bool * const` as an argument and return `void`.
     * @param ret A pointer to a boolean variable where the result of the
     * function call will be stored in.
     */
    template <typename function>
    __global__ void launch_test_kernel(function kernel_function,
                                       bool *const ret) {
        kernel_function(ret);
    }

    /**
     * @brief Launches a CUDA kernel for testing and returns the result.
     *
     * @note This function internally uses `test::launch_test_kernel` to launch
     * the kernel. The reason is, that `test_kernel` is often used to launch
     * lambda functions. Those can only be defined as `__device__` and not as
     * `__global__` functions.
     *
     * @attention The `kernel_function` can be a lambda function, but it is not
     * possible to have write access to the captured variables!
     *
     * @tparam function The function needs to take a `bool * const` as an
     * argument and return `void`.
     * @param kernel_function The `__device__` function to be launched as a
     * kernel. It must take a `bool * const` as an argument and return `void`.
     * @return The result of the kernel execution. `true` if the test passed,
     * `false` otherwise.
     */
    template <typename function>
    bool test_kernel(function kernel_function) {
        static_assert(std::is_invocable_r_v<void, function, bool *const>,
                      "kernel_function must be a function taking a 'bool * "
                      "const' and returning 'void'!");

        bool host_ret = true;
        bool *ret;
        core::checkCuda(cudaMalloc(&ret, sizeof(bool)));
        core::checkCuda(
            cudaMemcpy(ret, &host_ret, sizeof(bool), cudaMemcpyHostToDevice));

        launch_test_kernel<<<1, 1>>>(kernel_function, ret);
        core::checkCuda(cudaPeekAtLastError());

        core::checkCuda(
            cudaMemcpy(&host_ret, ret, sizeof(bool), cudaMemcpyDeviceToHost));
        core::checkCuda(cudaFree(ret));

        core::checkCuda(cudaDeviceSynchronize());
        return host_ret;
    }

}  // namespace test
