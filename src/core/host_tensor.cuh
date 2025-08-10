/**
 * @file host_tensor.cuh
 * @brief Implements the class Host_Tensor which manages CUDA memory on the
 * host.
 *
 * This file is part of the `core` namespace, which provides fundamental types,
 * utilities, and functions that are widely used across different modules of
 * the project.
 */

#pragma once

#include <algorithm>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda.cuh"
#include "device_tensor.cuh"
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
 * This specific file defines the Host_Tensor class.
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
     * @class Host_Tensor
     * @brief The Host_Tensor class manages CUDA memory on the host.
     */
    template <arithmetic_type dtype>
    class Host_Tensor {
      private:
        index_type _rank;
        index_type *_shape;
        index_type _batchsize;
        index_type _size;
        dtype *_data;

      public:
        /**
         * @brief Construct a new Host_Tensor object.
         *
         * This constructor creates a tensor object without requesting memory.
         */
        Host_Tensor()
            : _rank{0},
              _shape{nullptr},
              _batchsize{0},
              _size{0},
              _data{nullptr} {}

        /**
         * @brief Construct a new Host_Tensor object with a given shape.
         *
         * @param shape The shape of the tensor.
         */
        explicit Host_Tensor(const std::vector<index_type> &shape)
            : _rank{shape.size()},
              _batchsize{_rank == 0 ? 0 : shape[0]},
              _size{std::accumulate(shape.begin(), shape.end(), index_type(1),
                                    std::multiplies<index_type>())} {
            checkCuda(cudaMallocHost(&_shape, _rank * sizeof(index_type)));
            std::copy(shape.begin(), shape.end(), _shape);

            checkCuda(cudaMallocHost(&_data, _size * sizeof(dtype)));
        }

        /**
         * @brief Construct a new Host_Tensor object with a given shape.
         *
         * @note This constructor is needed for the case Host_Tensor({}).
         *
         * @param shape The shape of the tensor.
         */
        explicit Host_Tensor(const std::initializer_list<index_type> &shape)
            : Host_Tensor{std::vector<index_type>(shape)} {}

        /**
         * @brief Construct a new Host_Tensor object with a given shape but
         * without allocating memory.
         *
         * @note This constructor is useful when initializing, e.g., layers,
         * where we don't know the batchsize during the layer initialization.
         *
         * @param NO_ALLOC_type A tag type used for overload resolution to
         * indicate no allocation.
         * @param shape The shape (excluding the first axis) of the tensor.
         */
        explicit Host_Tensor(const NO_ALLOC_type /* unused */,
                             const std::vector<index_type> &shape)
            : _rank{1 + shape.size()},
              _batchsize{0},
              _size{std::accumulate(shape.begin(), shape.end(), index_type(1),
                                    std::multiplies<index_type>())},
              _data{nullptr} {
            checkCuda(cudaMallocHost(&_shape, _rank * sizeof(index_type)));
            _shape[0] = 0;
            std::copy(shape.begin(), shape.end(), _shape + 1);
        }

        /**
         * @brief Construct a new Host_Tensor object from a file.
         *
         * @param filename The file to read the tensor from. Excluding the
         * filetype.
         * @param filetype The filetype of the file.
         */
        explicit Host_Tensor(const std::string &filename,
                             const std::string filetype = "tensor")
            : Host_Tensor{} {
            load(filename, filetype);
        }

        /**
         * @brief Construct a new Host_Tensor object from another Host_Tensor.
         *
         * @param other The original tensor.
         */
        Host_Tensor(const Host_Tensor<dtype> &other) : Host_Tensor{} {
            copy_from(other);
        }

        /**
         * @brief Construct a new Host_Tensor object from another Host_Tensor
         * asynchronously.
         *
         * @param other The original tensor.
         * @param stream The CUDA stream used for copying.
         */
        explicit Host_Tensor(const Host_Tensor<dtype> &other,
                             const cudaStream_t &stream)
            : Host_Tensor{} {
            copy_from(other, stream);
        }

        /**
         * @brief Construct a new Host_Tensor object from a Device_Tensor.
         *
         * @param other The original tensor.
         */
        explicit Host_Tensor(const Device_Tensor<dtype> &other)
            : Host_Tensor{} {
            copy_from(other);
        }

        /**
         * @brief Construct a new Host_Tensor object from a Device_Tensor
         * asynchronously.
         *
         * @param other The original tensor.
         * @param stream The CUDA stream used for copying.
         */
        explicit Host_Tensor(const Device_Tensor<dtype> &other,
                             const cudaStream_t &stream)
            : Host_Tensor{} {
            copy_from(other, stream);
        }

        /**
         * @brief Move-Constructor.
         *
         * @param other The original tensor.
         */
        Host_Tensor(Host_Tensor<dtype> &&other) noexcept
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
        Host_Tensor &operator=(const Host_Tensor<dtype> &other) {
            return copy_from(other);
        }

        /**
         * @brief Move-Assignment Operator.
         *
         * @param other The original tensor.
         * @return The copied tensor.
         */
        Host_Tensor &operator=(Host_Tensor<dtype> &&other) noexcept {
            if (this == &other) return *this;

            std::swap(_rank, other._rank);
            std::swap(_shape, other._shape);
            std::swap(_batchsize, other._batchsize);
            std::swap(_size, other._size);
            std::swap(_data, other._data);

            return *this;
        }

        /**
         * @brief Destroy the Host_Tensor object.
         */
        ~Host_Tensor() noexcept {
            checkCuda(cudaFreeHost(_shape));
            checkCuda(cudaFreeHost(_data));
        }

        /**
         * @brief Saves the tensor to a file.
         *
         * @param filename The name of the file to write to. Excluding the
         * filetype.
         * @param filetype The filetype of the file.
         */
        void save(const std::string &filename,
                  const std::string filetype = "tensor") {
            std::string path;
            if (filetype != "")
                path = filename + "." + filetype;
            else
                path = filename;

            std::ofstream file(path);
            if (file.fail())
                throw std::runtime_error("Failed to open file: " + path);

            file << _rank;
            if (_rank) {
                file << "\n" << _shape[0];

                for (index_type idx = 1; idx < _rank; ++idx)
                    file << ' ' << _shape[idx];
            }

            if (_size) {
                file << "\n" << _data[0];
                for (index_type idx = 1; idx < _size; ++idx)
                    file << ' ' << _data[idx];
            }

            file.close();
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
        Host_Tensor &load(const std::string &filename,
                          const std::string filetype = "tensor") {
            std::string path;
            if (filetype != "")
                path = filename + "." + filetype;
            else
                path = filename;

            std::ifstream file(path);
            if (file.fail())
                throw std::runtime_error("Failed to open file: " + path);

            index_type new_rank;
            file >> new_rank;

            if (_rank != new_rank) {
                _rank = new_rank;
                checkCuda(cudaFreeHost(_shape));
                checkCuda(cudaMallocHost(&_shape, _rank * sizeof(index_type)));
            }

            for (index_type idx = 0; idx < _rank; ++idx) file >> _shape[idx];

            _batchsize = _rank == 0 ? 0 : _shape[0];

            index_type new_size =
                std::accumulate(_shape, _shape + _rank, index_type(1),
                                std::multiplies<index_type>());

            if (_data == nullptr || _size != new_size) {
                _size = new_size;
                checkCuda(cudaFreeHost(_data));
                checkCuda(cudaMallocHost(&_data, _size * sizeof(dtype)));
            }

            for (index_type idx = 0; idx < _size; ++idx) {
                std::string value_str;
                file >> value_str;

                long double value;

                if (value_str == "inf") {
                    value = std::numeric_limits<long double>::infinity();
                    continue;
                }

                else if (value_str == "-inf") {
                    value = -std::numeric_limits<long double>::infinity();
                    continue;
                }

                try {
                    value = std::stold(value_str);
                } catch (...) {
                    _data[idx] = std::numeric_limits<dtype>::quiet_NaN();
                    continue;
                }

                if constexpr (std::is_floating_point_v<dtype>) {
                    _data[idx] = value;
                } else {
                    if (value < std::numeric_limits<dtype>::lowest())
                        _data[idx] = std::numeric_limits<dtype>::lowest();
                    else if (value > std::numeric_limits<dtype>::max())
                        _data[idx] = std::numeric_limits<dtype>::max();
                    else
                        _data[idx] = value;
                }
            }

            file.close();

            return *this;
        }

        /**
         * @brief Makes a copy of another tensor and saves it into this one.
         *
         * @param other The tensor to copy from.
         * @return This tensor.
         */
        Host_Tensor &copy_from(const Host_Tensor<dtype> &other) {
            if (this == &other) return *this;

            if (_rank != other._rank) {
                _rank = other._rank;
                checkCuda(cudaFreeHost(_shape));
                checkCuda(cudaMallocHost(&_shape, _rank * sizeof(index_type)));
            }
            _batchsize = other._batchsize;
            checkCuda(cudaMemcpy(_shape, other._shape,
                                 _rank * sizeof(index_type),
                                 cudaMemcpyHostToHost));

            if (other._data == nullptr && _batchsize == 0) {  // NO_ALLOC case
                _size = other._size;
                checkCuda(cudaFreeHost(_data));
                _data = nullptr;
                return *this;
            }

            if (_data == nullptr || _size != other._size) {
                _size = other._size;
                checkCuda(cudaFreeHost(_data));
                checkCuda(cudaMallocHost(&_data, _size * sizeof(dtype)));
            }
            checkCuda(cudaMemcpy(_data, other._data, _size * sizeof(dtype),
                                 cudaMemcpyHostToHost));

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
        Host_Tensor &copy_from(const Host_Tensor<dtype> &other,
                               const cudaStream_t &stream) {
            if (this == &other) return *this;

            if (_rank != other._rank) {
                _rank = other._rank;
                checkCuda(cudaFreeHost(_shape));
                checkCuda(cudaMallocHost(&_shape, _rank * sizeof(index_type)));
            }
            _batchsize = other._batchsize;
            checkCuda(cudaMemcpyAsync(_shape, other._shape,
                                      _rank * sizeof(index_type),
                                      cudaMemcpyHostToHost, stream));

            if (other._data == nullptr && _batchsize == 0) {  // NO_ALLOC case
                _size = other._size;
                checkCuda(cudaFreeHost(_data));
                _data = nullptr;
                return *this;
            }

            if (_data == nullptr || _size != other._size) {
                _size = other._size;
                checkCuda(cudaFreeHost(_data));
                checkCuda(cudaMallocHost(&_data, _size * sizeof(dtype)));
            }
            checkCuda(cudaMemcpyAsync(_data, other._data, _size * sizeof(dtype),
                                      cudaMemcpyHostToHost, stream));

            return *this;
        }

        /**
         * @brief Copies from a Device_Tensor and saves it into this one.
         *
         * @param other The tensor to copy from.
         * @return This tensor.
         */
        Host_Tensor &copy_from(const Device_Tensor<dtype> &other) {
            if (_rank != other._rank) {
                _rank = other._rank;

                checkCuda(cudaFreeHost(_shape));
                checkCuda(cudaMallocHost(&_shape, _rank * sizeof(index_type)));
            }
            _batchsize = other._batchsize;
            checkCuda(cudaMemcpy(_shape, other._shape,
                                 _rank * sizeof(index_type),
                                 cudaMemcpyDeviceToHost));

            if (other._data == nullptr && _batchsize == 0) {  // NO_ALLOC case
                _size = other._size;
                checkCuda(cudaFreeHost(_data));
                _data = nullptr;
                return *this;
            }

            if (_data == nullptr || _size != other._size) {
                _size = other._size;
                checkCuda(cudaFreeHost(_data));
                checkCuda(cudaMallocHost(&_data, _size * sizeof(dtype)));
            }
            checkCuda(cudaMemcpy(_data, other._data, _size * sizeof(dtype),
                                 cudaMemcpyDeviceToHost));

            return *this;
        }

        /**
         * @brief Copies from a Device_Tensor and saves it into this one
         * asynchronously.
         *
         * @param other The tensor to copy from.
         * @param stream The CUDA stream used for copying.
         * @return This tensor.
         */
        Host_Tensor &copy_from(const Device_Tensor<dtype> &other,
                               const cudaStream_t &stream) {
            if (_rank != other._rank) {
                _rank = other._rank;

                checkCuda(cudaFreeHost(_shape));
                checkCuda(cudaMallocHost(&_shape, _rank * sizeof(index_type)));
            }
            _batchsize = other._batchsize;
            checkCuda(cudaMemcpyAsync(_shape, other._shape,
                                      _rank * sizeof(index_type),
                                      cudaMemcpyDeviceToHost, stream));

            if (other._data == nullptr && _batchsize == 0) {  // NO_ALLOC case
                _size = other._size;
                checkCuda(cudaFreeHost(_data));
                _data = nullptr;
                return *this;
            }

            if (_data == nullptr || _size != other._size) {
                _size = other._size;
                checkCuda(cudaFreeHost(_data));
                checkCuda(cudaMallocHost(&_data, _size * sizeof(dtype)));
            }
            checkCuda(cudaMemcpyAsync(_data, other._data, _size * sizeof(dtype),
                                      cudaMemcpyDeviceToHost, stream));

            return *this;
        }

        /**
         * @brief Friend Declaration of Device_Tensor<dtype>::copy_from.
         *
         * @param other This tensor.
         * @param stream The CUDA stream used for copying.
         * @return The copied tensor.
         */
        friend Device_Tensor<dtype> &Device_Tensor<dtype>::copy_from(
            const Host_Tensor<dtype> &other);
        friend Device_Tensor<dtype> &Device_Tensor<dtype>::copy_from(
            const Host_Tensor<dtype> &other, const cudaStream_t &stream);

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

                checkCuda(cudaFreeHost(_shape));
                checkCuda(cudaMallocHost(&_shape, _rank * sizeof(index_type)));
            }

            _batchsize = _rank == 0 ? 0 : shape[0];
            std::copy(shape.begin(), shape.end(), _shape);

            if (_data == nullptr || _size != new_size) {
                _size = new_size;
                checkCuda(cudaFreeHost(_data));
                checkCuda(cudaMallocHost(&_data, _size * sizeof(dtype)));
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
                _shape[0] = batchsize;

                checkCuda(cudaFreeHost(_data));
                checkCuda(cudaMallocHost(&_data, _size * sizeof(dtype)));
            }
        }

        /**
         * @brief Returns the data of the tensor.
         */
        const dtype *data() const { return _data; }

        /**
         * @brief Returns the data of the tensor.
         */
        dtype *data() { return _data; }

        /**
         * @brief Linear Access Operator.
         *
         * @param idx The index of the element we want to retrieve.
         * @return The retrieved element.
         */
        const dtype &operator[](const index_type &idx) const {
            return _data[idx];
        }

        /**
         * @brief Linear Access Operator.
         *
         * @param idx The index of the element we want to retrieve.
         * @return The retrieved element.
         */
        dtype &operator[](const index_type &idx) { return _data[idx]; }

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
         */
        const index_type *shape() const { return _shape; }

        /**
         * @brief Returns the shape of the tensor as a std::vector.
         */
        const std::vector<index_type> shape_vec() const {
            return std::vector<index_type>(_shape, _shape + _rank);
        }

        /**
         * @brief Returns the shape (excluding the first axis) of the tensor as
         * a std::vector.
         *
         * @param NO_BATCHSIZE_type A tag type used for overload resolution.
         */
        const std::vector<index_type> shape_vec(
            const NO_BATCHSIZE_type /* unused */) const {
            return std::vector<index_type>(_shape + 1, _shape + _rank);
        }

        /**
         * @brief Shape Access Function.
         *
         * @param idx The index of the element we want to retrieve.
         * @return The element we want to retrieve.
         */
        const index_type &shape(const index_type &idx) const {
            return _shape[idx];
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
            std::ostringstream oss;

            oss << "{";
            oss << "rank: " << _rank << ", ";

            if (_rank) {
                oss << "shape: [" << _shape[0];

                for (index_type idx = 1; idx < _rank; ++idx)
                    oss << ", " << _shape[idx];

                oss << "], ";
            } else
                oss << "[], ";

            oss << "size: " << _size << ", ";
            if (_size) {
                oss << "data: [" << _data[0];

                if (_size < max_show)
                    for (index_type idx = 1; idx < _size; ++idx)
                        oss << ", " << _data[idx];

                else {
                    for (index_type idx = 1; idx < max_show / 2; ++idx)
                        oss << ", " << _data[idx];

                    oss << ", ...";

                    for (index_type idx = _size - max_show / 2; idx < _size;
                         ++idx)
                        oss << ", " << _data[idx];
                }

                oss << "]}";
            } else
                oss << "[]}";

            return oss.str();
        }

        /**
         * @brief Returns the memory footprint of the object on GPU.
         */
        const Mem_Size mem_size(const bool /* unused */ = false) const {
            return {0, 0};
        }
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
                             const Host_Tensor<dtype> &tensor) {
        out << tensor.str();
        return out;
    }

}  // namespace core
