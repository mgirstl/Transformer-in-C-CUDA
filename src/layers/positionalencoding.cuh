/**
 * @file positionalencoding.cuh
 * @brief Implements the PositionalEncoding class.
 *
 * This file is part of the `layers` namespace, which implements differentiable
 * layers.
 */

#pragma once

#include <cmath>
#include <vector>

#include "../core.cuh"
#include "../optimizers/none.cuh"
#include "dropout.cuh"
#include "layer.cuh"

/**
 * @internal
 *
 * @namespace layers
 * @brief Implements differentiable layers.
 *
 * This specific file defines the PositionalEncoding class.
 *
 * @endinternal
 */
namespace layers {

    /**
     * @brief This kernel pre-computes the shift used in the PositionalEncoding
     * layer.
     *
     * @param tensor The output tensor where the results will be stored in. It
     * has to have the shape @f$ [K, M] @f$.
     */
    template <core::float_type data_type>
    __global__ void positionalencoding_initialize(
        core::Kernel_Tensor<data_type> tensor) {
        const core::index_type row = blockIdx.x * blockDim.x + threadIdx.x;
        const core::index_type row_stride = gridDim.x * blockDim.x;
        const core::index_type col = blockIdx.y * blockDim.y + threadIdx.y;
        const core::index_type col_stride = gridDim.y * blockDim.y;

        const data_type factor = -log(10000.0) / tensor.shape(1);
        for (core::index_type r = row; r < tensor.shape(0); r += row_stride) {
            for (core::index_type c = col; c < tensor.shape(1);
                 c += col_stride) {
                const data_type div_term = exp(2 * factor * (c / 2));
                const data_type angle = r * div_term + M_PI / 2 * (c % 2);
                tensor[r * tensor.shape(1) + c] = sin(angle);
            }
        }
    }

    /**
     * @brief This kernel adds to the input tensor the pre-computed shifts and
     * saves them in the output tensor.
     *
     * @param input The input tensor containing the values to be processed. It
     * has to have the shape @f$ [K, N, M] @f$.
     * @param output The tensor to store the end result in. It has to have the
     * shape @f$ [K, N, M] @f$.
     * @param pe The pre-computed shift values. It has to have the shape
     * @f$ [K, M] @f$.
     */
    template <core::float_type data_type>
    __global__ void positionalencoding_forward_add(
        const core::Kernel_Tensor<data_type> input,
        core::Kernel_Tensor<data_type> output,
        const core::Kernel_Tensor<data_type> pe) {
        const core::index_type thread_x = blockIdx.x * blockDim.x + threadIdx.x;
        const core::index_type x_stride = gridDim.x * blockDim.x;
        const core::index_type thread_y = blockIdx.y * blockDim.y + threadIdx.y;
        const core::index_type y_stride = gridDim.y * blockDim.y;
        const core::index_type thread_z = blockIdx.z * blockDim.z + threadIdx.z;
        const core::index_type z_stride = gridDim.z * blockDim.z;

        for (core::index_type x = thread_x; x < output.shape(0);
             x += x_stride) {
            for (core::index_type y = thread_y; y < output.shape(1);
                 y += y_stride) {
                for (core::index_type z = thread_z; z < output.shape(2);
                     z += z_stride) {
                    output[x * output.sample_size() + y * output.shape(2) + z] =
                        input[x * output.sample_size() + y * output.shape(2) +
                              z] +
                        pe[x * pe.shape(1) + z];
                }
            }
        }
    }

    /**
     * @class PositionalEncoding
     * @brief The Positional Encoding layer.
     *
     * The Positional Encoding layer adds positional information to the input
     * tensor. This is crucial for models like the Transformer, which do not
     * have any inherent notion of the order of the input sequence.
     *
     * The positional encodings are computed as:
     * @f[
     *     PE_{(x, 2k)} = \sin\left(
     *                        \frac{x}{10000^{2z/d_{\mathrm{embedding}}}}
     *                    \right)
     * @f]
     * @f[
     *     PE_{(x, 2k+1)} = \cos\left(
     *                          \frac{x}{10000^{2z/d_{\mathrm{embedding}}}}
     *                      \right)
     * @f]
     * where @f$ x @f$ is the position of the sample in the current batch and
     * @f$ z @f$ is the position in the embedding dimension.
     * @f$ d_{\mathrm{embedding}} @f$ is the size of the embedding dimension.
     *
     * @note This layer is in training mode after initialization. This means
     * in this particular case that in the forward pass dropout is applied. In
     * contrast, in evaluation mode the dropout behaves like the identity
     * function.
     *
     * @tparam data_type The data type of the tensor elements.
     */
    template <core::float_type data_type>
    class PositionalEncoding final
        : public Layer<data_type, data_type, optimizers::None> {
      private:
        core::Device &device;
        core::Device_Tensor<data_type> pe;
        core::Device_Tensor<data_type> output;
        Dropout<data_type> dropout;

      public:
        /**
         * @brief Construct a new PositionalEncoding object.
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param shape The shape (excluding the first axis) of the input/output
         * tensors, i.e., `[sequence_length, embedding_dim]`.
         * @param dropout The probability that an element in the dropout layer
         * is set to zero.
         * @param max_batchsize The maximum batchsize to be expected.
         * @param stream The CUDA stream used for launching the kernels.
         */
        PositionalEncoding(core::Device &device,
                           const std::vector<core::index_type> &shape,
                           const data_type dropout = 0.1,
                           const core::index_type max_batchsize = 5000,
                           const cudaStream_t stream = core::default_stream)
            : device{device},
              pe{{max_batchsize, shape.back()}},
              output{core::NO_ALLOC, shape},
              dropout{device, shape, dropout, stream} {
            positionalencoding_initialize<data_type>
                <<<device.blocks_2D(pe.batchsize(), shape.back()),
                   device.threads_2D(), 0, stream>>>(pe);
            core::checkCuda(cudaPeekAtLastError());
        }

        /**
         * @brief Compute the forward pass of the layer.
         *
         * @note This function will also apply Dropout in training mode.
         *
         * @param input The input tensor.
         * @param stream The CUDA stream used for launching the kernels.
         * @return The output tensor.
         */
        const core::Device_Tensor<data_type> &forward(
            const core::Device_Tensor<data_type> &input,
            const cudaStream_t stream = core::default_stream) override {
            output.rebatchsize(input.batchsize(), stream);

            core::index_type n = input.batchsize() + pe.sample_size();
            n = (input.size() + n - 1) / n;
            positionalencoding_forward_add<data_type>
                <<<device.blocks_3D(input.batchsize(), n, pe.sample_size()),
                   device.threads_3D(), 0, stream>>>(input, output, pe);
            core::checkCuda(cudaPeekAtLastError());

            return dropout.forward(output, stream);
        }

        /**
         * @brief Compute the backward pass of the layer.
         *
         * @param error The error tensor from the next layer.
         * @param stream The CUDA stream used for launching the kernels.
         * @return The gradient with respect to the input.
         */
        const core::Device_Tensor<data_type> &backward(
            const core::Device_Tensor<data_type> &error,
            const cudaStream_t stream = core::default_stream) override {
            return dropout.backward(error, stream);
        }

        /**
         * @brief Get the output of the forward pass.
         */
        const core::Device_Tensor<data_type> &get_output() const override {
            return dropout.get_output();
        }

        /**
         * @brief Get the gradient in respect to the input.
         */
        const core::Device_Tensor<data_type> &get_input_gradient()
            const override {
            return dropout.get_input_gradient();
        }

        /**
         * @brief Get the pre-computed positional encodings.
         */
        const core::Device_Tensor<data_type> &get_pe() const { return pe; }

        /**
         * @brief Get the dropout layer.
         */
        const Dropout<data_type> &get_dropout() const { return dropout; }

        /**
         * @brief Sets the layer into training mode.
         *
         * In training mode, dropout is applied in the forward pass.
         *
         * This layer is initially set to training mode.
         */
        void train() override { dropout.train(); }

        /**
         * @brief Sets the layer into evaluation mode.
         *
         * In evaluation mode, the dropout behaves like the identity function.
         *
         * This layer is initially set to training mode.
         */
        void eval() override { dropout.eval(); }

        /**
         * @brief Returns the memory footprint of the object on GPU.
         */
        const core::Mem_Size mem_size(
            const bool /* unused */ = false) const override {
            core::Mem_Size size;
            size += dropout.mem_size();
            size += pe.mem_size(true);
            size += output.mem_size();
            return size;
        }
    };

}  // namespace layers
