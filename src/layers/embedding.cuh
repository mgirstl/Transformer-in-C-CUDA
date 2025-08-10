/**
 * @file embedding.cuh
 * @brief Implements the Embedding class.
 *
 * This file is part of the `layers` namespace, which implements differentiable
 * layers.
 */

#pragma once

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "../core.cuh"
#include "layer.cuh"

/**
 * @internal
 *
 * @namespace layers
 * @brief Implements differentiable layers.
 *
 * This specific file defines the Embedding class.
 *
 * @endinternal
 */
namespace layers {

    /**
     * @brief This kernel calculates the forward pass of the embedding layer.
     *
     * This kernel performs the matrix multiplication between the one-hot
     * encoded input and the weights matrix in a sparse manner. Additionally,
     * it prepares the tensor `input_onehot`, which is used in the matrix
     * multiplication during the backward pass.
     *
     * @param input The input tensor containing the indices to extract. E.g.,
     * in the case of a natural language processing task, these could be the
     * token-ids. It must have the shape `[batchsize, sequence_length]`.
     * @param input_onehot The input_onehot tensor to store the input in one-hot
     * encoded form in. It must have the shape
     * `[batchsize, sequence_length, num_embeddings]`.
     * @param output The output tensor to store the results of the matrix
     * multiplication in. It must have the shape
     * `[batchsize, sequence_length, embedding_dim]`.
     * @param weights The weights tensor contains the weights that are
     * multiplied with the one-hot encoded input tensor. It must have the shape
     * `[num_embeddings, embedding_dim]`.
     * @param scale An additional scaling factor that gets multiplied with the
     * output.
     */
    template <core::integer_type input_type, core::float_type output_type>
    __global__ void embedding_forward(
        const core::Kernel_Tensor<input_type> input,
        core::Kernel_Tensor<output_type> input_onehot,
        core::Kernel_Tensor<output_type> output,
        const core::Kernel_Tensor<output_type> weights,
        const output_type scale) {
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
                    const core::index_type idx = input[x * input.shape(1) + y];
                    input_onehot[x * input_onehot.sample_size() +
                                 y * input_onehot.shape(2) + idx] = 1;
                    output[x * output.sample_size() + y * output.shape(2) + z] =
                        scale * weights[idx * weights.shape(1) + z];
                }
            }
        }
    }

    /**
     * @class Embedding
     * @brief The Embedding layer.
     *
     * The embedding layer transforms input indices into dense vectors of
     * fixed size, effectively converting categorical data into continuous
     * numerical representations that can be used by neural networks. This
     * conversion is done by first one-hot encoding these indices and then
     * multiplying them with a learnable weights matrix.
     *
     * @note This layer is in training mode after initialization. This means
     * in this particular case that the backward pass will update the trainable
     * parameters.
     *
     * @note For convenience, the elements of the `input_onehot` tensor have the
     * `output_type`type instead of `core::mask_type`. The reason for this is
     * that `core::matrix_multiplication` is only defined for floating-point
     * values.
     *
     * @tparam input_type The data type of the input tensor elements.
     * @tparam output_type The data type of the output tensor elements.
     * @tparam optimizer_type The optimizer used for the parameter updates.
     */
    template <core::integer_type input_type, core::float_type output_type,
              template <typename> class optimizer_type>
    class Embedding final
        : public Layer<input_type, output_type, optimizer_type> {
      private:
        core::Device &device;
        input_type num_embeddings;
        output_type scale;
        core::Device_Tensor<output_type> output;
        core::Device_Tensor<output_type> input_onehot;
        core::Device_Tensor<output_type> weights;
        core::Device_Tensor<output_type> weights_gradient;
        optimizer_type<output_type> optimizer;
        core::Device_Tensor<input_type> null_tensor;
        bool training;

      public:
        /**
         * @brief Construct a new Embedding object.
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param input_shape The shape (excluding the first axis) of the input
         * tensors, i.e., `[sequence_length]`.
         * @param output_shape The shape (excluding the first axis) of the
         * output tensors, i.e., `[sequence_length, embedding_dim]`.
         * @param num_embeddings The size of the dictionary, i.e., this value
         * corresponds to one larger than the largest value in the input tensor.
         * @param optimizer_kw This will be passed to the optimizer. See the
         * documentation of the used optimizer to see which parameters are
         * possible.
         * @param scale A scaling factor which gets multiplied to the output of
         * the layer.
         * @param stream The CUDA stream used for launching the kernels.
         *
         * @note `embedding_dim` is the size of each embedding vector.
         * `sequence_length` is the number of tokens in one sample (/sequence).
         */
        Embedding(core::Device &device,
                  const std::vector<core::index_type> &input_shape,
                  const std::vector<core::index_type> &output_shape,
                  const input_type num_embeddings,
                  const std::unordered_map<std::string, output_type>
                      &optimizer_kw = {},
                  const output_type scale = 1,
                  const cudaStream_t stream = core::default_stream)
            : device{device},
              num_embeddings{num_embeddings},
              scale{scale},
              output{core::NO_ALLOC, output_shape},
              input_onehot{core::NO_ALLOC, {input_shape[0], num_embeddings}},
              weights{{num_embeddings, output_shape.back()}},
              weights_gradient{{num_embeddings, output_shape.back()}},
              optimizer{device,
                        {num_embeddings, output_shape.back()},
                        optimizer_kw,
                        stream},
              training{true} {
            auto generator = device.curand_generator(stream);
            output_type std = sqrt(2.0 / num_embeddings);
            core::checkCuda(
                core::generate_normal<output_type>(generator, weights, 0, std));
            core::checkCuda(cudaPeekAtLastError());
        }

        /**
         * @brief Compute the forward pass of the layer.
         *
         * @param input The input tensor.
         * @param stream The CUDA stream used for launching the kernels.
         * @return The output tensor.
         */
        const core::Device_Tensor<output_type> &forward(
            const core::Device_Tensor<input_type> &input,
            const cudaStream_t stream = core::default_stream) override {
            output.rebatchsize(input.batchsize(), stream);
            input_onehot.rebatchsize(input.batchsize(), stream);

            core::checkCuda(cudaMemsetAsync(
                input_onehot.data(), 0,
                input_onehot.size() * sizeof(output_type), stream));
            core::checkCuda(cudaPeekAtLastError());

            embedding_forward<input_type, output_type>
                <<<device.blocks_3D(input.batchsize(), input.sample_size(),
                                    weights.sample_size()),
                   device.threads_3D(), 0, stream>>>(input, input_onehot,
                                                     output, weights, scale);
            core::checkCuda(cudaPeekAtLastError());

            return output;
        }

        /**
         * @brief Compute the backward pass of the layer.
         *
         * @note This function will also update the parameters if the layer is
         * in training mode.
         *
         * @param error The error tensor from the next layer.
         * @param stream The CUDA stream used for launching the kernels.
         * @return An empty tensor.
         *
         * @attention This function returns an empty tensor instead of the
         * `input_gradient`. The reason for this is that the `input_gradient` is
         * not defined for an integer input.
         */
        const core::Device_Tensor<input_type> &backward(
            const core::Device_Tensor<output_type> &error,
            const cudaStream_t stream = core::default_stream) override {
            core::matrix_multiplication<output_type>(
                device.cublas_handle(stream), scale, input_onehot,
                num_embeddings, CUBLAS_OP_T, error, weights.sample_size(),
                CUBLAS_OP_N, 0.0, weights_gradient, num_embeddings,
                input_onehot.size() / num_embeddings, weights.sample_size());

            if (training) optimizer.update(weights, weights_gradient, stream);

            return null_tensor;
        }

        /**
         * @brief Get the output of the forward pass.
         */
        const core::Device_Tensor<output_type> &get_output() const override {
            return output;
        }

        /**
         * @brief Get the gradient with respect to the input.
         *
         * @attention This function returns an empty tensor instead of the
         * `input_gradient`. The reason for this is that the `input_gradient` is
         * not defined for an integer input.
         */
        const core::Device_Tensor<input_type> &get_input_gradient()
            const override {
            return null_tensor;
        }

        /**
         * @brief Get the weights.
         */
        const core::Device_Tensor<output_type> &get_weights() const {
            return weights;
        }

        /**
         * @brief Sets the layer into training mode.
         *
         * In training mode, the parameter update step will be performed when
         * calling the backward pass of the layer.
         *
         * This layer is initially set to training mode.
         */
        void train() override { training = true; }

        /**
         * @brief Sets the layer into evaluation mode.
         *
         * In evaluation mode, the parameter update step will not be performed
         * when calling the backward pass of the layer.
         *
         * This layer is initially set to training mode.
         */
        void eval() override { training = false; }

        /**
         * @brief Returns the number of trainable parameters of the layer.
         */
        core::index_type parameters() const override { return weights.size(); }

        /**
         * @brief Returns the memory footprint of the object on GPU.
         */
        const core::Mem_Size mem_size(
            const bool /* unused */ = false) const override {
            core::Mem_Size size;
            size += weights.mem_size(true);
            size += weights_gradient.mem_size(true);
            size += optimizer.mem_size();
            size += output.mem_size();
            size += input_onehot.mem_size();
            size += null_tensor.mem_size(true);
            return size;
        }

      private:
        /**
         * @brief Saves the parameters of the layer to a file.
         *
         * @param path The base path to write the parameters to.
         * @param filetype The filetype of the saved files.
         */
        void _save(const std::string &path,
                   const std::string &filetype) override {
            weights.save(path + "_weights", filetype);
        }

        /**
         * @brief Loads the parameters of the layer from a file.
         *
         * @param path The base path to read the parameters from.
         * @param filetype The filetype of the saved files.
         * @param stream The CUDA stream used for copying from the host.
         */
        void _load(const std::string &path, const std::string &filetype,
                   const cudaStream_t &stream) override {
            weights.load(path + "_weights", filetype, stream);
        }
    };

}  // namespace layers
