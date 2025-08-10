/**
 * @file cuda.cuh
 * @brief Contains functions and macros related to CUDA error checking.
 *
 * This file is part of the `core` namespace, which provides fundamental types,
 * utilities, and functions that are widely used across different modules of
 * the project.
 */

#pragma once

#include <cublas_v2.h>
#include <curand.h>

#include <cstdlib>
#include <iostream>
#include <string>

/**
 * @namespace core
 * @brief Provides core facilities used throughout the entire codebase.
 *
 * The `core` namespace contains fundamental types, utilities, and functions
 * that are widely used across different modules of the project.
 *
 * @internal
 * This specific file defines functions and macros related to CUDA error
 * checking.
 * @endinternal
 *
 * @note The checkCuda macro is defined inside this namespace for consistency,
 * but it is available globally because it's a macro.
 */
namespace core {

// Compiled with CUDA and --DDEBUG flag:
#if defined(DEBUG) && defined(__CUDACC__)

/**
 * @brief Prints the CUDA/cuBLAS/cuRAND error string if an error has
 * occurred.
 *
 * @attention Please use as core::checkCuda if not used inside the core
 * namespace.
 *
 * @param error The error which occurred.
 * @return The original error.
 */
#define checkCuda(val) checkCudaErrors((val), #val, __FILE__, __LINE__)

    /**
     * @brief Prints the CUDA error string if an error has occurred and
     * exits after 10 errors.
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
    inline cudaError_t checkCudaErrors(cudaError_t error,
                                       char const *const func,
                                       const char *const file, int const line) {
        static int error_count = 0;

        if (error != cudaSuccess) {
            std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(error)
                      << "\n\tat " << file << ":" << line << " '" << func
                      << "'\n";
            ++error_count;

            if (error_count >= 10) {
                std::cerr << "Too many errors. Exiting..." << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        return error;
    }

    /**
     * @brief Prints the cuBLAS status string if an error has occurred and
     * exits after 10 errors.
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
    inline cublasStatus_t checkCudaErrors(cublasStatus_t error,
                                          char const *const func,
                                          const char *const file,
                                          int const line) {
        static int error_count = 0;

        if (error != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "CUBLAS Runtime Error: "
                      << cublasGetStatusString(error) << "\n\tat " << file
                      << ":" << line << " '" << func << "'\n";
            ++error_count;

            if (error_count >= 10) {
                std::cerr << "Too many errors. Exiting..." << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        return error;
    }

    /**
     * @brief Converts a cuRAND error into a human readable string.
     *
     * Source:
     * https://stackoverflow.com/a/40704198
     * CC BY-SA 3.0 - Copyright (c) 2016 Vince Varga
     *
     * @param error The error which occurred.
     * @return The corresponding error string.
     */
    inline std::string curandGetStatusString(curandStatus_t error) {
        switch (error) {
            case CURAND_STATUS_SUCCESS:
                return "CURAND_STATUS_SUCCESS";
            case CURAND_STATUS_VERSION_MISMATCH:
                return "CURAND_STATUS_VERSION_MISMATCH";
            case CURAND_STATUS_NOT_INITIALIZED:
                return "CURAND_STATUS_NOT_INITIALIZED";
            case CURAND_STATUS_ALLOCATION_FAILED:
                return "CURAND_STATUS_ALLOCATION_FAILED";
            case CURAND_STATUS_TYPE_ERROR:
                return "CURAND_STATUS_TYPE_ERROR";
            case CURAND_STATUS_OUT_OF_RANGE:
                return "CURAND_STATUS_OUT_OF_RANGE";
            case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
                return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
            case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
                return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
            case CURAND_STATUS_LAUNCH_FAILURE:
                return "CURAND_STATUS_LAUNCH_FAILURE";
            case CURAND_STATUS_PREEXISTING_FAILURE:
                return "CURAND_STATUS_PREEXISTING_FAILURE";
            case CURAND_STATUS_INITIALIZATION_FAILED:
                return "CURAND_STATUS_INITIALIZATION_FAILED";
            case CURAND_STATUS_ARCH_MISMATCH:
                return "CURAND_STATUS_ARCH_MISMATCH";
            case CURAND_STATUS_INTERNAL_ERROR:
                return "CURAND_STATUS_INTERNAL_ERROR";
        }
        return "CURAND_STATUS_UNKNOWN";
    }

    /**
     * @brief Prints the cuRAND status string if an error has occurred and
     * exits after 10 errors.
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
    inline curandStatus_t checkCudaErrors(curandStatus_t error,
                                          char const *const func,
                                          const char *const file,
                                          int const line) {
        static int error_count = 0;

        if (error != CURAND_STATUS_SUCCESS) {
            std::cerr << "CURAND Runtime Error: "
                      << curandGetStatusString(error) << "\n\tat " << file
                      << ":" << line << " '" << func << "'\n";
            ++error_count;

            if (error_count >= 10) {
                std::cerr << "Too many errors. Exiting..." << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        return error;
    }

// Compiled without CUDA or without --DDEBUG flag:
#else

/**
 * @brief Defines the checkCuda macro to do nothing when compiled
 * without CUDA or without --DDEBUG flag.
 *
 * @attention Please use as core::checkCuda if not used inside the core
 * namespace.
 *
 * @param error The error which occurred.
 * @return The original error.
 */
#define checkCuda(val) checkCudaErrors((val))

    /**
     * @brief Defines the checkCudaErrors function to do nothing without
     * CUDA or without --DDEBUG flag.
     *
     * @note This function is needed because the checkCuda macro only
     * replaces 'checkCuda' in 'core::checkCuda'. If we would define
     * checkCuda as '#define checkCuda(val) (val)' we would replace
     * 'core::checkCuda(val)' with 'core::(val)' which is not valid.
     * Hence, we need to define 'core::checkCudaErrors' here.
     *
     * @param error The error which occurred.
     * @return The original error.
     */
    template <typename error_type>
    inline error_type checkCudaErrors(error_type error) {
        return error;
    }

#endif

}  // namespace core
