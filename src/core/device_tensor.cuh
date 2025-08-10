/**
 * @file device_tensor.cuh
 * @brief Implements the class Device_Tensor which manages CUDA memory on the
 * device.
 *
 * This file is part of the `core` namespace, which provides fundamental types,
 * utilities, and functions that are widely used across different modules of
 * the project.
 */
#pragma once

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "cuda.cuh"
#include "host_tensor.cuh"
#include "kernel_tensor.cuh"
#include "mem_size.cuh"
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
 * This specific file defines the Device_Tensor class.
 *
 * @endinternal
 */
namespace core {

    /**
     * @internal
     *
     * @class Host_Tensor
     * @brief Forward declaration of Host_Tensor.
     * @see host_tensor.cuh
     *
     * @endinternal
     */
    template <arithmetic_type dtype>
    class Host_Tensor;

    /**
     * @internal
     *
     * @class Kernel_Tensor
     * @brief Forward declaration of Kernel_Tensor.
     * @see kernel_tensor.cuh
     *
     * @endinternal
     */
    template <arithmetic_type dtype>
    class Kernel_Tensor;

    /**
     * @class Device_Tensor
     * @brief The Device_Tensor class manages CUDA memory on the device.
     *
     * @note This class only contains the memory management for the device data.
     * On the device itself, e.g., for calculations, the class Kernel_Tensor is
     * used.
     */
    template <arithmetic_type dtype>
    class Device_Tensor {
      private:
        index_type _rank;
        index_type *_shape;
        index_type _batchsize;
        index_type _size;
        dtype *_data;

      public:
        /**
         * @brief Construct a new Device_Tensor object.
         *
         * This constructor creates a tensor object without requesting memory.
         */
        Device_Tensor()
            : _rank{0},
              _shape{nullptr},
              _batchsize{0},
              _size{0},
              _data{nullptr} {}

        /**
         * @brief Construct a new Device_Tensor object with a given shape.
         *
         * @param shape The shape of the tensor.
         */
        explicit Device_Tensor(const std::vector<index_type> &shape)
            : _rank{shape.size()},
              _batchsize{_rank == 0 ? 0 : shape[0]},
              _size{std::accumulate(shape.begin(), shape.end(), index_type(1),
                                    std::multiplies<index_type>())} {
            index_type *host_shape;
            checkCuda(cudaMallocHost(&host_shape, _rank * sizeof(index_type)));
            checkCuda(cudaMalloc(&_shape, _rank * sizeof(index_type)));

            std::copy(shape.begin(), shape.end(), host_shape);
            checkCuda(cudaMemcpy(_shape, host_shape, _rank * sizeof(index_type),
                                 cudaMemcpyHostToDevice));

            checkCuda(cudaFreeHost(host_shape));

            checkCuda(cudaMalloc(&_data, _size * sizeof(dtype)));
        }

        /**
         * @brief Construct a new Device_Tensor object with a given shape.
         *
         * @note This constructor is needed for the case Device_Tensor({}).
         *
         * @param shape The shape of the tensor.
         */
        explicit Device_Tensor(const std::initializer_list<index_type> &shape)
            : Device_Tensor{std::vector<index_type>(shape)} {}

        /**
         * @brief Construct a new Device_Tensor object with a given shape but
         * without allocating memory.
         *
         * @note This constructor is useful when initializing, e.g., layers,
         * where we don't know the batchsize during the layer initialization.
         *
         * @param NO_ALLOC_type A tag type used for overload resolution to
         * indicate no allocation.
         * @param shape The shape (excluding the first axis) of the tensor.
         */
        explicit Device_Tensor(const NO_ALLOC_type /* unused */,
                               const std::vector<index_type> &shape)
            : _rank{1 + shape.size()},
              _batchsize{0},
              _size{std::accumulate(shape.begin(), shape.end(), index_type(1),
                                    std::multiplies<index_type>())},
              _data{nullptr} {
            index_type *host_shape;
            checkCuda(cudaMallocHost(&host_shape, _rank * sizeof(index_type)));
            checkCuda(cudaMalloc(&_shape, _rank * sizeof(index_type)));

            host_shape[0] = 0;
            std::copy(shape.begin(), shape.end(), host_shape + 1);
            checkCuda(cudaMemcpy(_shape, host_shape, _rank * sizeof(index_type),
                                 cudaMemcpyHostToDevice));

            checkCuda(cudaFreeHost(host_shape));
        }

        /**
         * @brief Construct a new Device_Tensor object from a file.
         *
         * @param filename The file to read the tensor from. Excluding the
         * filetype.
         * @param filetype The filetype of the file.
         */
        explicit Device_Tensor(const std::string &filename,
                               const std::string &filetype = "tensor")
            : Device_Tensor{} {
            load(filename, filetype);
        }

        /**
         * @brief Construct a new Device_Tensor object from a file
         * asynchronously.
         *
         * @note Assumes filetype = "tensor" in load(filename, stream) call.
         *
         * @param filename The file to read the tensor from. Excluding the
         * filetype.
         * @param stream The CUDA stream used for copying from the host.
         */
        explicit Device_Tensor(const std::string &filename,
                               const cudaStream_t &stream)
            : Device_Tensor{} {
            load(filename, stream);
        }

        /**
         * @brief Construct a new Device_Tensor object from a file
         * asynchronously.
         *
         * @param filename The file to read the tensor from. Excluding the
         * filetype.
         * @param filetype The filetype of the file.
         * @param stream The CUDA stream used for copying from the host.
         */
        explicit Device_Tensor(const std::string &filename,
                               const std::string &filetype,
                               const cudaStream_t &stream)
            : Device_Tensor{} {
            load(filename, filetype, stream);
        }

        /**
         * @brief Construct a new Device_Tensor object from another
         * Device_Tensor.
         *
         * @param other The original tensor.
         */
        Device_Tensor(const Device_Tensor<dtype> &other) : Device_Tensor{} {
            copy_from(other);
        }

        /**
         * @brief Construct a new Device_Tensor object from another
         * Device_Tensor asynchronously.
         *
         * @param other The original tensor.
         * @param stream The CUDA stream used for copying.
         */
        explicit Device_Tensor(const Device_Tensor<dtype> &other,
                               const cudaStream_t &stream)
            : Device_Tensor{} {
            copy_from(other, stream);
        }

        /**
         * @brief Construct a new Device_Tensor object from a Host_Tensor.
         *
         * @param other The original tensor.
         */
        explicit Device_Tensor(const Host_Tensor<dtype> &other)
            : Device_Tensor{} {
            copy_from(other);
        }

        /**
         * @brief Construct a new Device_Tensor object from a Host_Tensor
         * asynchronously.
         *
         * @param other The original tensor.
         * @param stream The CUDA stream used for copying from the host.
         */
        explicit Device_Tensor(const Host_Tensor<dtype> &other,
                               const cudaStream_t &stream)
            : Device_Tensor{} {
            copy_from(other, stream);
        }

        /**
         * @brief Move-Constructor.
         *
         * @param other The original tensor.
         */
        Device_Tensor(Device_Tensor<dtype> &&other) noexcept
            : _rank{other._rank},
              _shape{other._shape},
              _batchsize{other._batchsize},
              _size{other._size},
              _data{other._data} {
            other._rank = 0;
            other._shape = nullptr;
            other._batchsize = 0;
            other._size = 0;
            other._data = nullptr;
        }

        /**
         * @brief Copy-Assignment Operator.
         *
         * @param other The tensor to copy from.
         * @return The copied tensor.
         */
        Device_Tensor &operator=(const Device_Tensor<dtype> &other) {
            return copy_from(other);
        }

        /**
         * @brief Move-Assignment Operator.
         *
         * @param other The original tensor.
         * @return The copied tensor.
         */
        Device_Tensor &operator=(Device_Tensor<dtype> &&other) noexcept {
            if (this == &other) return *this;

            std::swap(_rank, other._rank);
            std::swap(_shape, other._shape);
            std::swap(_batchsize, other._batchsize);
            std::swap(_size, other._size);
            std::swap(_data, other._data);

            return *this;
        }

        /**
         * @brief Destroy the Device_Tensor object.
         */
        ~Device_Tensor() noexcept {
            checkCuda(cudaFree(_shape));
            checkCuda(cudaFree(_data));
        }

        /**
         * @brief Saves the tensor to a file.
         *
         * @param filename The name of the file to write to. Excluding the
         * filetype.
         * @param filetype The filetype of the file.
         */
        void save(const std::string &filename,
                  const std::string &filetype = "tensor") {
            Host_Tensor<dtype> tensor(*this);
            tensor.save(filename, filetype);
        }

        /**
         * @brief Loads the tensor from a file.
         *
         * @note Values which can't be read are stored as `NAN`. If the tensor
         * is an integer tensor then values outside of the representable range
         * will be clipped to the representable range.
         *
         * @param filename The file to read the tensor from. Excluding the
         * filetype.
         * @param filetype The filetype of the file.
         * @return This tensor.
         */
        Device_Tensor &load(const std::string &filename,
                            const std::string &filetype = "tensor") {
            Host_Tensor<dtype> tensor(filename, filetype);
            return copy_from(tensor);
        }

        /**
         * @brief Loads the tensor from a file asynchronously.
         *
         * @note Values which can't be read are stored as `NAN`. If the tensor
         * is an integer tensor then values outside of the representable range
         * will be clipped to the representable range.
         *
         * @note Assumes filetype = "tensor" in Host_Tensor(filename) call.
         *
         * @param filename The file to read the tensor from. Excluding the
         * filetype.
         * @param stream The CUDA stream used for copying from the host.
         */
        Device_Tensor &load(const std::string &filename,
                            const cudaStream_t &stream) {
            Host_Tensor<dtype> tensor(filename);
            return copy_from(tensor, stream);
        }

        /**
         * @brief Loads the tensor from a file asynchronously.
         *
         * @note Values which can't be read are stored as `NAN`. If the tensor
         * is an integer tensor then values outside of the representable range
         * will be clipped to the representable range.
         *
         * @note Assumes filetype = "tensor" in Host_Tensor(filename) call.
         *
         * @param filename The file to read the tensor from. Excluding the
         * filetype.
         * @param filetype The filetype of the file.
         * @param stream The CUDA stream used for copying from the host.
         */
        Device_Tensor &load(const std::string &filename,
                            const std::string &filetype,
                            const cudaStream_t &stream) {
            Host_Tensor<dtype> tensor(filename, filetype);
            return copy_from(tensor, stream);
        }

        /**
         * @brief Makes a copy of another tensor and saves it into this one.
         *
         * @param other The tensor to copy from.
         * @return This tensor.
         */
        Device_Tensor &copy_from(const Device_Tensor<dtype> &other) {
            if (this == &other) return *this;

            if (_rank != other._rank) {
                _rank = other._rank;

                checkCuda(cudaFree(_shape));
                checkCuda(cudaMalloc(&_shape, _rank * sizeof(index_type)));
            }
            _batchsize = other._batchsize;
            checkCuda(cudaMemcpy(_shape, other._shape,
                                 _rank * sizeof(index_type),
                                 cudaMemcpyDeviceToDevice));

            if (other._data == nullptr && _batchsize == 0) {  // NO_ALLOC case
                _size = other._size;
                checkCuda(cudaFree(_data));
                _data = nullptr;
                return *this;
            }

            if (_data == nullptr || _size != other._size) {
                _size = other._size;
                checkCuda(cudaFree(_data));
                checkCuda(cudaMalloc(&_data, _size * sizeof(dtype)));
            }
            checkCuda(cudaMemcpy(_data, other._data, _size * sizeof(dtype),
                                 cudaMemcpyDeviceToDevice));

            return *this;
        }

        /**
         * @brief Makes a copy of another tensor and saves it into this one
         * asynchronously.
         *
         * @param other The tensor to copy from.
         * @param stream The CUDA stream used for copying.
         * @return This tensor.
         */
        Device_Tensor &copy_from(const Device_Tensor<dtype> &other,
                                 const cudaStream_t &stream) {
            if (this == &other) return *this;

            if (_rank != other._rank) {
                _rank = other._rank;

                checkCuda(cudaFreeAsync(_shape, stream));
                checkCuda(cudaMallocAsync(&_shape, _rank * sizeof(index_type),
                                          stream));
            }
            _batchsize = other._batchsize;
            checkCuda(cudaMemcpyAsync(_shape, other._shape,
                                      _rank * sizeof(index_type),
                                      cudaMemcpyDeviceToDevice, stream));

            if (other._data == nullptr && _batchsize == 0) {  // NO_ALLOC case
                _size = other._size;
                checkCuda(cudaFreeAsync(_data, stream));
                _data = nullptr;
                return *this;
            }

            if (_data == nullptr || _size != other._size) {
                _size = other._size;
                checkCuda(cudaFreeAsync(_data, stream));
                checkCuda(
                    cudaMallocAsync(&_data, _size * sizeof(dtype), stream));
            }
            checkCuda(cudaMemcpyAsync(_data, other._data, _size * sizeof(dtype),
                                      cudaMemcpyDeviceToDevice, stream));

            return *this;
        }

        /**
         * @brief Copies from a Host_Tensor and saves it into this one.
         *
         * @param other The tensor to copy from.
         * @return This tensor.
         */
        Device_Tensor &copy_from(const Host_Tensor<dtype> &other) {
            if (_rank != other._rank) {
                _rank = other._rank;

                checkCuda(cudaFree(_shape));
                checkCuda(cudaMalloc(&_shape, _rank * sizeof(index_type)));
            }
            _batchsize = other._batchsize;
            checkCuda(cudaMemcpy(_shape, other._shape,
                                 _rank * sizeof(index_type),
                                 cudaMemcpyHostToDevice));

            if (other._data == nullptr && _batchsize == 0) {  // NO_ALLOC case
                _size = other._size;
                checkCuda(cudaFree(_data));
                _data = nullptr;
                return *this;
            }

            if (_data == nullptr || _size != other._size) {
                _size = other._size;
                checkCuda(cudaFree(_data));
                checkCuda(cudaMalloc(&_data, _size * sizeof(dtype)));
            }
            checkCuda(cudaMemcpy(_data, &other[0], _size * sizeof(dtype),
                                 cudaMemcpyHostToDevice));

            return *this;
        }

        /**
         * @brief Copies from a Host_Tensor and saves it into this one
         * asynchronously.
         *
         * @param other The tensor to copy from.
         * @param stream The CUDA stream used for copying.
         * @return This tensor.
         */
        Device_Tensor &copy_from(const Host_Tensor<dtype> &other,
                                 const cudaStream_t &stream) {
            if (_rank != other._rank) {
                _rank = other._rank;

                checkCuda(cudaFreeAsync(_shape, stream));
                checkCuda(cudaMallocAsync(&_shape, _rank * sizeof(index_type),
                                          stream));
            }
            _batchsize = other._batchsize;
            checkCuda(cudaMemcpyAsync(_shape, other._shape,
                                      _rank * sizeof(index_type),
                                      cudaMemcpyHostToDevice, stream));

            if (other._data == nullptr && _batchsize == 0) {  // NO_ALLOC case
                _size = other._size;
                checkCuda(cudaFreeAsync(_data, stream));
                _data = nullptr;
                return *this;
            }

            if (_data == nullptr || _size != other._size) {
                _size = other._size;
                checkCuda(cudaFreeAsync(_data, stream));
                checkCuda(
                    cudaMallocAsync(&_data, _size * sizeof(dtype), stream));
            }
            checkCuda(cudaMemcpyAsync(_data, &other[0], _size * sizeof(dtype),
                                      cudaMemcpyHostToDevice, stream));

            return *this;
        }

        /**
         * @brief Friend Declaration of Host_Tensor<dtype>::copy_from.
         *
         * @param other This tensor.
         * @param stream The CUDA stream used for copying.
         * @return The copied tensor.
         */
        friend Host_Tensor<dtype> &Host_Tensor<dtype>::copy_from(
            const Device_Tensor<dtype> &other);
        friend Host_Tensor<dtype> &Host_Tensor<dtype>::copy_from(
            const Device_Tensor<dtype> &other, const cudaStream_t &stream);

        /**
         * @brief Changes the shape of the tensor.
         *
         * @attention This function does not check if the product of the shape
         * before and after updating is the same! If the number of elements
         * change, this will lead to data loss!
         *
         * @param shape The new shape.
         */
        void reshape(const std::vector<index_type> &shape) {
            index_type new_rank = shape.size();
            index_type new_size =
                std::accumulate(shape.begin(), shape.end(), index_type(1),
                                std::multiplies<index_type>());

            if (_rank != new_rank) {
                _rank = new_rank;

                checkCuda(cudaFree(_shape));
                checkCuda(cudaMalloc(&_shape, _rank * sizeof(index_type)));
            }

            _batchsize = _rank == 0 ? 0 : shape[0];

            index_type *host_shape;
            checkCuda(cudaMallocHost(&host_shape, _rank * sizeof(index_type)));
            std::copy(shape.begin(), shape.end(), host_shape);
            checkCuda(cudaMemcpy(_shape, host_shape, _rank * sizeof(index_type),
                                 cudaMemcpyHostToDevice));
            checkCuda(cudaFreeHost(host_shape));

            if (_data == nullptr || _size != new_size) {
                _size = new_size;
                checkCuda(cudaFree(_data));
                checkCuda(cudaMalloc(&_data, _size * sizeof(dtype)));
            }
        }

        /**
         * @brief Changes the batchsize of the tensor.
         *
         * @attention If the number of elements change, this will lead to data
         * loss!
         *
         * @param batchsize The new batchsize.
         */
        void rebatchsize(const index_type batchsize) {
            if (batchsize != 0 && _batchsize != batchsize) {
                _size = batchsize * sample_size();
                _batchsize = batchsize;
                checkCuda(cudaFree(_data));
                checkCuda(cudaMalloc(&_data, _size * sizeof(dtype)));
                checkCuda(cudaMemcpy(_shape, &_batchsize, sizeof(index_type),
                                     cudaMemcpyHostToDevice));
            }
        }

        /**
         * @brief Changes the batchsize of the tensor.
         *
         * @attention If the number of elements change, this will lead to data
         * loss!
         *
         * @param batchsize The new batchsize.
         * @param stream The CUDA stream used for allocation and copying.
         */
        void rebatchsize(const index_type batchsize,
                         const cudaStream_t &stream) {
            if (_batchsize != batchsize) {
                _size = batchsize * sample_size();
                _batchsize = batchsize;
                checkCuda(cudaFreeAsync(_data, stream));
                checkCuda(
                    cudaMallocAsync(&_data, _size * sizeof(dtype), stream));
                checkCuda(cudaMemcpyAsync(_shape, &_batchsize,
                                          sizeof(index_type),
                                          cudaMemcpyHostToDevice, stream));
            }
        }

        /**
         * @brief Returns the data of the tensor.
         *
         * @note This points to memory on the device.
         */
        const dtype *data() const { return _data; }

        /**
         * @brief Returns the data of the tensor.
         *
         * @note This points to memory on the device.
         */
        dtype *data() { return _data; }

        /**
         * @brief Returns the batchsize of the tensor.
         */
        const index_type &batchsize() const { return _batchsize; }

        /**
         * @brief Returns the sample_size of the tensor.
         *
         * The sample size is the size of the tensor excluding the batchsize.
         */
        index_type sample_size() const {
            if (_batchsize == 0) return _size;
            return _size / _batchsize;
        }

        /**
         * @brief Returns the number of elements of the tensor.
         */
        const index_type &size() const { return _size; }

        /**
         * @brief Returns the rank of the tensor.
         */
        const index_type &rank() const { return _rank; }

        /**
         * @brief Returns the shape of the tensor.
         *
         * @note This points to memory on the device.
         */
        const index_type *shape() const { return _shape; }

        /**
         * @brief Returns the shape of the tensor as a std::vector.
         */
        const std::vector<index_type> shape_vec() const {
            index_type *host_shape;
            checkCuda(cudaMallocHost(&host_shape, _rank * sizeof(index_type)));
            checkCuda(cudaMemcpy(host_shape, _shape, _rank * sizeof(index_type),
                                 cudaMemcpyDeviceToHost));

            auto _shape_vec =
                std::vector<index_type>(host_shape, host_shape + _rank);
            checkCuda(cudaFreeHost(host_shape));

            return _shape_vec;
        }

        /**
         * @brief Returns the shape (excluding the first axis) of the tensor as
         * a std::vector.
         *
         * @param NO_BATCHSIZE_type A tag type used for overload resolution.
         */
        const std::vector<index_type> shape_vec(
            const NO_BATCHSIZE_type /* unused */) const {
            index_type *host_shape;
            checkCuda(cudaMallocHost(&host_shape, _rank * sizeof(index_type)));
            checkCuda(cudaMemcpy(host_shape, _shape, _rank * sizeof(index_type),
                                 cudaMemcpyDeviceToHost));

            auto _shape_vec =
                std::vector<index_type>(host_shape + 1, host_shape + _rank);
            checkCuda(cudaFreeHost(host_shape));

            return _shape_vec;
        }

        /**
         * @brief This function returns a string containing the tensor's rank,
         * shape, size, and data.
         *
         * This function returns a string containing the tensor's rank, shape,
         * size, and data. If the tensor's size is less than max_show, all
         * elements are inserted. Otherwise, the first `max_show / 2` and last
         * `max_show / 2` elements are printed, with an ellipsis (...)
         * indicating omitted elements.
         *
         * @param max_show The maximum number of elements to show without
         * truncation.
         * @return The created string.
         */
        std::string str(const index_type max_show = 7) const {
            return Host_Tensor(*this).str(max_show);
        }

        /**
         * @brief Returns the memory footprint of the object on GPU.
         *
         * @param fixed True if the tensor does not change its batchsize during
         * its lifetime. False if it is expected to change its batchsize during
         * its lifetime.
         * @return The memory footprint.
         */
        const Mem_Size mem_size(const bool fixed = false) const {
            if (fixed)
                return {_rank * sizeof(index_type) + _size * sizeof(dtype), 0};
            else
                return {_rank * sizeof(index_type),
                        sample_size() * sizeof(dtype)};
        }

        /**
         * @brief Friend Declaration of Kernel_Tensor.
         */
        friend class Kernel_Tensor<dtype>;
    };

    /**
     * @brief This function prints the tensor's rank, shape, size, and data to
     * the provided output stream.
     *
     * This function overloads the << operator for the Host_Tensor class. It
     * prints the tensor's rank, shape, size, and data to the provided output
     * stream using the tensor.str() method.
     *
     * @param out The output stream to which the tensor information is printed.
     * @param tensor The tensor whose information is printed.
     * @return The same output stream passed in the `out` parameter, allowing
     * for chaining of output operations.
     */
    template <arithmetic_type dtype>
    std::ostream &operator<<(std::ostream &out,
                             const Device_Tensor<dtype> &tensor) {
        out << tensor.str();
        return out;
    }

}  // namespace core
