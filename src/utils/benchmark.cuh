/**
 * @file benchmark.cuh
 * @brief Implements some useful functions for benchmarking code.
 *
 * This file is part of the `utils` namespace, which implements utility
 * functions and classes.
 */

#pragma once

#include <chrono>
#include <functional>
#include <iostream>
#include <string>

#include "../core.cuh"

/**
 * @internal
 *
 * @namespace utils
 * @brief Namespace for utility functions and classes.
 *
 * This specific file defines some useful functions for benchmarking code.
 *
 * @endinternal
 */
namespace utils {

    /**
     * @brief Runs a function and measures its runtime.
     *
     * This function first runs the provided function `func` for a specified
     * number of warmup steps to ensure any initial overhead is accounted for.
     * It then measures the runtime of the function over a specified number of
     * steps.
     *
     * @param name The name of the benchmark.
     * @param func The function to be benchmarked.
     * @param steps The number of steps to measure the runtime.
     * @param warmup_steps The number of warmup steps to run before measuring.
     */
    void bench(const std::string name, std::function<void(void)> func,
               const core::index_type steps = 100,
               const core::index_type warmup_steps = 10) {
        for (core::index_type idx = 0; idx < warmup_steps; ++idx) {
            func();
            core::checkCuda(cudaGetLastError());
        }

        core::checkCuda(cudaDeviceSynchronize());
        auto start = std::chrono::system_clock::now();

        for (core::index_type idx = 0; idx < steps; ++idx) {
            func();
            core::checkCuda(cudaGetLastError());
        }

        core::checkCuda(cudaDeviceSynchronize());
        auto end = std::chrono::system_clock::now();

        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();

        std::cout << ">>> " << name << std::endl;
        std::cout << "Total Runtime [ms]: " << duration / 1000.0 << std::endl;
        std::cout << "Runtime per Iteration [ms]: "
                  << duration / (1000.0 * steps) << std::endl;
        std::cout << "Steps: " << steps << std::endl;
    }

    /**
     * @brief Get the current memory usage in bytes.
     */
    size_t get_memory_usage() {
        size_t free_byte;
        size_t total_byte;
        core::checkCuda(cudaMemGetInfo(&free_byte, &total_byte));
        return total_byte - free_byte;
    }

}  // namespace utils
