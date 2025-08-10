/**
 * @file kernel_tensor.cuh
 * @brief Implements the class Kernel_Tensor which is used for calculations on
 * the host. It does not contain memory management.
 *
 * This file is part of the `core` namespace, which provides fundamental types,
 * utilities, and functions that are widely used across different modules of
 * the project.
 */
#pragma once

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
 * This specific file defines the Kernel_Tensor class.
 *
 * @endinternal
 */
namespace core {

    /**
     * @internal
     *
     * @class Device_Tensor
     * @brief Forward declaration of Device_Tensor.
     * @see device_tensor.cuh
     *
     * @endinternal
     */
    template <arithmetic_type dtype>
    class Device_Tensor;

    /**
     * @class Kernel_Tensor
     * @brief The Kernel_Tensor class is used for calculations on the device.
     *
     * @note This class does not contain memory management. The only way to
     * create a Kernel_Tensor is by casting a Device_Tensor to a Kernel_Tensor.
     *
     * @warning The Kernel_Tensor class does not maintain const correctness. Due
     * to the friendship between Device_Tensor and Kernel_Tensor, it is possible
     * to modify members of a const Device_Tensor through a Kernel_Tensor.
     */
    template <arithmetic_type dtype>
    class Kernel_Tensor {
      private:
        index_type _rank;
        index_type *_shape;
        index_type _batchsize;
        index_type _size;
        dtype *_data;

      public:
        /**
         * @brief Construct a new Kernel_Tensor object from a Device_Tensor.
         *
         * @param other The original tensor.
         */
        Kernel_Tensor(const Device_Tensor<dtype> &tensor)
            : _rank{tensor._rank},
              _shape{tensor._shape},
              _batchsize{tensor._batchsize},
              _size{tensor._size},
              _data{tensor._data} {}

        /**
         * @brief Returns the data of the tensor.
         */
        __host__ __device__ const dtype *data() const { return _data; }

        /**
         * @brief Returns the data of the tensor.
         */
        __device__ dtype *data() { return _data; }

        /**
         * @brief Linear Access Operator.
         *
         * @param idx The index of the element we want to retrieve.
         * @return The retrieved element.
         */
        __device__ const dtype &operator[](const index_type idx) const {
            return _data[idx];
        }

        /**
         * @brief Linear Access Operator.
         *
         * @param idx The index of the element we want to retrieve.
         * @return The retrieved element.
         */
        __device__ dtype &operator[](const index_type idx) {
            return _data[idx];
        }

        /**
         * @brief Returns the batchsize of the tensor.
         */
        __host__ __device__ const index_type &batchsize() const {
            return _batchsize;
        }

        /**
         * @brief Returns the sample_size of the tensor.
         *
         * The sample size is the size of the tensor excluding the batchsize.
         */
        __host__ __device__ index_type sample_size() const {
            return _size / _batchsize;
        }

        /**
         * @brief Returns the number of elements of the tensor.
         */
        __host__ __device__ const index_type &size() const { return _size; }

        /**
         * @brief Returns the rank of the tensor.
         */
        __host__ __device__ const index_type &rank() const { return _rank; }

        /**
         * @brief Returns the shape of the tensor.
         */
        __host__ __device__ const index_type *shape() const { return _shape; }

        /**
         * @brief Shape Access Function.
         *
         * @param idx The index of the element we want to retrieve.
         * @return The element we want to retrieve.
         */
        __device__ const index_type &shape(const index_type &idx) const {
            return _shape[idx];
        }
    };

}  // namespace core
