/**
 * @file dataloader.cuh
 * @brief Implements the DataLoader class.
 *
 * This file is part of the `utils` namespace, which implements utility
 * functions and classes.
 */

#pragma once

#include <algorithm>
#include <limits>
#include <random>
#include <string>

#include "../core.cuh"

/**
 * @internal
 *
 * @namespace utils
 * @brief Namespace for utility functions and classes.
 *
 * This specific file defines the DataLoader class.
 *
 * @endinternal
 */
namespace utils {

    /**
     * @brief This kernel copies a batch of samples in a specified order.
     *
     * @param complete A tensor containing the whole dataset. It has to have the
     * shape @f$ [N, *] @f$.
     * @param current A tensor to save the current batch in. It has to have the
     * shape @f$ [M, *] @f$. with @f$ M \leq N @f$.
     * @param indices A tensor specifying in which order elements from the
     * complete dataset should be copied into the current batch. It has to have
     * the shape @f$ [N] @f$. Each value needs to be unique and between 0 and
     * @f$ N-1 @f$.
     * @param start The number of samples already copied. This value indicates
     * the starting position in the indices array from which elements should be
     * copied. This value should be between 0 and @f$ N-1 @f$.
     *
     * @note The @f$ * @f$ symbol denotes any number of dimensions and must be
     * the same in all tensors where it is used.
     */
    template <core::arithmetic_type data_type>
    __global__ void dataloader_copy_batch(
        const core::Kernel_Tensor<data_type> complete,
        core::Kernel_Tensor<data_type> current,
        const core::Kernel_Tensor<core::index_type> indices,
        const core::index_type start) {
        const core::index_type row = blockIdx.x * blockDim.x + threadIdx.x;
        const core::index_type row_stride = gridDim.x * blockDim.x;
        const core::index_type col = blockIdx.y * blockDim.y + threadIdx.y;
        const core::index_type col_stride = gridDim.y * blockDim.y;

        const core::index_type n_rows = current.shape(0);
        const core::index_type n_cols = current.size() / n_rows;

        for (core::index_type r = row; r < n_rows; r += row_stride) {
            const core::index_type r_wrapped =
                (row + start) % complete.shape(0);
            for (core::index_type c = col; c < n_cols; c += col_stride)
                current[r * n_cols + c] =
                    complete[indices[r_wrapped] * n_cols + c];
        }
    }

    /**
     * @class DataLoader
     * @brief A class to handle data loading for training and testing.
     *
     * This class manages loading a specified dataset of training samples and
     * training targets from disk. Additionally, it handles counting the number
     * of iterations and epochs, and can shuffle the dataset for training.
     *
     * @tparam data_type The data type of the input data.
     * @tparam label_type The data type of the target data.
     */
    template <core::arithmetic_type data_type, core::arithmetic_type label_type>
    class DataLoader {
      private:
        const core::Device &device;
        bool shuffle;
        core::index_type batchsize;
        core::index_type max_epoch;
        core::index_type max_batch;
        core::index_type max_iteration;
        core::index_type current_epoch;
        core::index_type current_batch;
        core::index_type current_iteration;
        bool new_epoch;
        bool first_epoch;
        bool first_batch;
        bool first_iteration;
        std::mt19937 generator;
        core::Device_Tensor<data_type> complete_data;
        core::Device_Tensor<label_type> complete_target;
        core::Device_Tensor<data_type> current_data;
        core::Device_Tensor<label_type> current_target;
        core::index_type total_num_samples;
        core::Host_Tensor<core::index_type> host_indices;
        core::Device_Tensor<core::index_type> device_indices;

      public:
        /**
         * @brief Constructs a new DataLoader object.
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param data_path The path to the input data file.
         * @param target_path The path to the target data file.
         * @param batchsize The size of each batch.
         * @param shuffle Whether to shuffle the data at the beginning of each
         * epoch.
         * @param max_epoch The maximum number of epochs.
         * @param max_iteration The maximum number of iterations.
         * @param max_batch The maximum number of batches.
         * @param stream The CUDA stream used for copying from host to device.
         */
        DataLoader(const core::Device &device, const std::string &data_path,
                   const std::string &target_path,
                   const core::index_type batchsize, bool shuffle,
                   core::index_type max_epoch =
                       std::numeric_limits<core::index_type>::max(),
                   core::index_type max_iteration =
                       std::numeric_limits<core::index_type>::max(),
                   core::index_type max_batch =
                       std::numeric_limits<core::index_type>::max(),
                   const cudaStream_t stream = core::default_stream)
            : device{device},
              shuffle{shuffle},
              batchsize{batchsize},
              max_epoch{max_epoch},
              max_batch{max_batch},
              max_iteration{max_iteration},
              current_epoch{0},
              current_batch{0},
              current_iteration{0},
              new_epoch{true},
              first_epoch{true},
              first_batch{true},
              first_iteration{true},
              generator{0},
              complete_data{data_path, stream},
              complete_target{target_path, stream},
              current_data{core::NO_ALLOC,
                           complete_data.shape_vec(core::NO_BATCHSIZE)},
              current_target{core::NO_ALLOC,
                             complete_target.shape_vec(core::NO_BATCHSIZE)},
              total_num_samples{complete_data.batchsize()},
              host_indices{{total_num_samples}},
              device_indices{{total_num_samples}} {
            current_data.rebatchsize(batchsize, stream);
            current_target.rebatchsize(batchsize, stream);

            for (core::index_type idx = 0; idx < host_indices.size(); ++idx)
                host_indices[idx] = idx;
            device_indices.copy_from(host_indices, stream);
        }

        /**
         * @brief Loads the next batch of data.
         *
         * @param stream The CUDA stream used for launching the kernels.
         * @return true If there is still data to load.
         * @return false If the specified number of epochs, number of
         * iterations, or number of total batches is reached.
         */
        bool next(const cudaStream_t stream = core::default_stream) {
            if (first_iteration)
                first_iteration = false;
            else
                ++current_iteration;

            if (new_epoch) {
                if (shuffle) {
                    std::shuffle(host_indices.data(),
                                 host_indices.data() + host_indices.size(),
                                 generator);
                    device_indices.copy_from(host_indices, stream);
                }

                if (first_epoch)
                    first_epoch = false;
                else
                    ++current_epoch;

                first_batch = true;
                current_batch = 0;
            }

            if (first_batch)
                first_batch = false;
            else
                ++current_batch;

            core::index_type start = current_batch * batchsize;

            dataloader_copy_batch<data_type>
                <<<device.blocks_2D(batchsize, current_data.sample_size()),
                   device.threads_2D(), 0, stream>>>(
                    complete_data, current_data, device_indices, start);
            core::checkCuda(cudaPeekAtLastError());

            dataloader_copy_batch<label_type>
                <<<device.blocks_2D(batchsize, current_target.sample_size()),
                   device.threads_2D(), 0, stream>>>(
                    complete_target, current_target, device_indices, start);
            core::checkCuda(cudaPeekAtLastError());

            new_epoch = start + batchsize >= total_num_samples;

            return current_epoch < max_epoch &&
                   current_iteration < max_iteration &&
                   current_batch < max_batch;
        }

        /**
         * @brief This function resets all internal counters.
         */
        void reset() {
            current_epoch = 0;
            current_batch = 0;
            current_iteration = 0;
            new_epoch = true;
            first_epoch = true;
            first_batch = true;
            first_iteration = true;
        }

        /**
         * @brief Get the current epoch number.
         */
        const core::index_type &epoch() const { return current_epoch; }

        /**
         * @brief Get the current batch number.
         */
        const core::index_type &batch() const { return current_batch; }

        /**
         * @brief Get the current iteration number.
         */
        const core::index_type &iteration() const { return current_iteration; }

        /**
         * @brief Get the total number of samples in the dataset.
         */
        const core::index_type &num_samples() const {
            return total_num_samples;
        }

        /**
         * @brief Get the current batch of input data.
         */
        const core::Device_Tensor<data_type> &data() const {
            return current_data;
        }

        /**
         * @brief Get the current batch of target data.
         */
        const core::Device_Tensor<label_type> &target() const {
            return current_target;
        }

        /**
         * @brief Returns the memory footprint of the object on GPU.
         */
        const core::Mem_Size mem_size(const bool /* unused */ = false) const {
            core::Mem_Size size;
            size += complete_data.mem_size(true);
            size += complete_target.mem_size(true);
            size += device_indices.mem_size(true);
            size += current_data.mem_size();
            size += current_target.mem_size();
            return size;
        }
    };

}  // namespace utils
