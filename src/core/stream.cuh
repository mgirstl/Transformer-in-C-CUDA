/**
 * @file stream.cuh
 * @brief Implements the class Stream which is a wrapper around `cudaStream_t`.
 *
 * This file is part of the `core` namespace, which provides fundamental types,
 * utilities, and functions that are widely used across different modules of
 * the project.
 */
#pragma once

#include <utility>

#include "cuda.cuh"

/**
 * @internal
 *
 * @namespace core
 * @brief Provides core facilities used throughout the entire codebase.
 *
 * The `core` namespace contains fundamental types, utilities, and functions
 * that are widely used across different modules of the project.
 *
 * This specific file defines the Stream class.
 *
 * @endinternal
 */
namespace core {

    /**
     * @brief Saves `cudaStreamDefault` into a variable so that it can be used
     * as an lvalue of type `cudaStream_t`. This is necessary because it is
     * defined inside CUDA via a macro.
     */
    constexpr cudaStream_t default_stream = cudaStreamDefault;

    /**
     * @class Stream
     * @brief A wrapper class for CUDA streams with dependency management.
     */
    class Stream {
      private:
        cudaStream_t stream;

      public:
        /**
         * @brief Constructor to create a new CUDA stream.
         */
        Stream() { checkCuda(cudaStreamCreate(&stream)); }

        /**
         * @brief Destructor to destroy the CUDA stream.
         */
        ~Stream() {
            if (stream != 0) checkCuda(cudaStreamDestroy(stream));
        }

        // Delete copy constructor
        Stream(const Stream &) = delete;

        // Delete copy assignment operator
        Stream &operator=(const Stream &) = delete;

        // Move constructor
        Stream(Stream &&other) noexcept : stream(other.stream) {
            other.stream = default_stream;
        }

        // Move assignment operator
        Stream &operator=(Stream &&other) noexcept {
            if (this != &other) std::swap(stream, other.stream);
            return *this;
        }

        /**
         * @brief Let the current Stream wait for another `cudaStream_t` to
         * complete.
         *
         * @param other The `cudaStream_t` to wait for.
         */
        void wait(const cudaStream_t other) {
            cudaEvent_t event;
            checkCuda(cudaEventCreate(&event));
            checkCuda(cudaEventRecord(event, other));
            checkCuda(cudaStreamWaitEvent(stream, event, 0));
            checkCuda(cudaEventDestroy(event));
        }

        /**
         * @brief Let the current Stream wait for another Stream to complete.
         *
         * @param other The Stream to wait for.
         */
        void wait(const Stream &other) { wait(cudaStream_t(other)); }

        /**
         * @brief Let the other `cudaStream_t` wait for this Stream to complete.
         *
         * @param other The `cudaStream_t` which should wait.
         */
        void make_wait(const cudaStream_t other) const {
            cudaEvent_t event;
            checkCuda(cudaEventCreate(&event));
            checkCuda(cudaEventRecord(event, stream));
            checkCuda(cudaStreamWaitEvent(other, event, 0));
            checkCuda(cudaEventDestroy(event));
        }

        /**
         * @brief Let the other Stream wait for this Stream to complete.
         *
         * @param other The Stream which should wait.
         */
        void make_wait(const Stream &other) const {
            make_wait(cudaStream_t(other));
        }

        /**
         * @brief Cast operator to `cudaStream_t`.
         */
        operator cudaStream_t() { return stream; }

        /**
         * @brief Cast operator to `cudaStream_t`.
         */
        operator cudaStream_t() const { return stream; }
    };

}  // namespace core
