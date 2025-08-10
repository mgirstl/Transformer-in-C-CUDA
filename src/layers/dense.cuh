/**
 * @file dense.cuh
 * @brief Implements the Dense class.
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
 * This specific file defines the Dense class.
 *
 * @endinternal
 */
namespace layers {

    /**
     * @brief This kernel copies the bias into the output matrix.
     *
     * The copying takes place so that each row of the output matrix gets
     * the same bias vector. This makes it possible to combine the matrix
     * multiplication of weights and input with the adding of the bias
     * afterwards using the `core::matrix_multiplication` function.
     *
     * @param bias The bias tensor containing the values to copy. It has to have
     * the shape @f$ [N] @f$.
     * @param output The output tensor to store the values in.  It has to have
     * the shape @f$ [*, N] @f$.
     *
     * @note The @f$ * @f$ symbol denotes any number of dimensions and must be
     * the same in all tensors where it is used.
     */
    template <core::float_type data_type>
    __global__ void dense_forward_copy_bias(
        const core::Kernel_Tensor<data_type> bias,
        core::Kernel_Tensor<data_type> output) {
        const core::index_type row = blockIdx.x * blockDim.x + threadIdx.x;
        const core::index_type row_stride = gridDim.x * blockDim.x;
        const core::index_type col = blockIdx.y * blockDim.y + threadIdx.y;
        const core::index_type col_stride = gridDim.y * blockDim.y;

        const core::index_type num_cols = bias.size();
        const core::index_type num_rows = output.size() / num_cols;

        for (core::index_type r = row; r < num_rows; r += row_stride)
            for (core::index_type c = col; c < num_cols; c += col_stride)
                output[r * num_cols + c] = bias[c];
    }

    /**
     * @class Dense
     * @brief The Dense layer.
     *
     * The Dense layer computes:
     * @f[ f(\vec{x}) = w\cdot\vec{x} + \vec{b} @f]
     * for each sample @f$ \vec{x} @f$. Here @f$ w @f$ represents the weights
     * matrix and @f$ \vec{b} @f$ the bias vector.
     *
     * @note This layer is in training mode after initialization. This means
     * in this particular case that the backward pass will update the trainable
     * parameters.
     *
     * @tparam data_type The data type of the tensor elements.
     * @tparam optimizer_type The optimizer used for the parameter updates.
     */
    template <core::float_type data_type,
              template <typename> class optimizer_type>
    class Dense final : public Layer<data_type, data_type, optimizer_type> {
      private:
        core::Device &device;
        bool use_bias;
        core::index_type in_features;
        core::index_type out_features;
        core::Device_Tensor<data_type> weights;
        core::Device_Tensor<data_type> weights_gradient;
        optimizer_type<data_type> weights_optimizer;
        core::Device_Tensor<data_type> bias;
        core::Device_Tensor<data_type> bias_gradient;
        core::Device_Tensor<data_type> one;
        optimizer_type<data_type> bias_optimizer;
        core::Device_Tensor<data_type> input;
        core::Device_Tensor<data_type> output;
        core::Device_Tensor<data_type> input_gradient;
        core::Device_Tensor<data_type> null_tensor;
        bool training;

      public:
        /**
         * @brief Construct a new Dense object.
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param input_shape The shape (excluding the first axis) of the input
         * tensors, i.e., @f$ [*, M] @f$.
         * @param output_shape The shape (excluding the first axis) of the
         * output tensors, i.e., @f$ [*, N] @f$.
         * @param optimizer_kw Will be passed to the optimizer. See the
         * documentation of the used optimizer to see which parameters are
         * possible.
         * @param use_bias If set to False, the layer will not learn an additive
         * bias.
         * @param stream The CUDA stream used for launching the kernels.
         *
         * @note The @f$ * @f$ symbol denotes any number of values but they need
         * to be the same in all shapes where it is used.
         */
        Dense(
            core::Device &device,
            const std::vector<core::index_type> &input_shape,
            const std::vector<core::index_type> &output_shape,
            const std::unordered_map<std::string, data_type> &optimizer_kw = {},
            const bool use_bias = true,
            const cudaStream_t stream = core::default_stream)
            : device{device},
              use_bias{use_bias},
              in_features{input_shape.back()},
              out_features{output_shape.back()},
              weights{{in_features, out_features}},
              weights_gradient{{in_features, out_features}},
              weights_optimizer{
                  device, {in_features, out_features}, optimizer_kw, stream},
              bias{use_bias ? std::vector<core::index_type>{out_features}
                            : std::vector<core::index_type>{}},
              bias_gradient{use_bias
                                ? std::vector<core::index_type>{out_features}
                                : std::vector<core::index_type>{}},
              one{core::NO_ALLOC, {}},
              bias_optimizer{device,
                             use_bias
                                 ? std::vector<core::index_type>{out_features}
                                 : std::vector<core::index_type>{},
                             optimizer_kw, stream},
              input{core::NO_ALLOC, input_shape},
              output{core::NO_ALLOC, output_shape},
              input_gradient{core::NO_ALLOC, input_shape},
              training{true} {
            auto generator = device.curand_generator(stream);
            data_type std = sqrt(2.0 / in_features);

            core::checkCuda(
                core::generate_normal<data_type>(generator, weights, 0, std));
            core::checkCuda(cudaPeekAtLastError());

            if (use_bias)
                core::checkCuda(
                    core::generate_normal<data_type>(generator, bias, 0, std));
            core::checkCuda(cudaPeekAtLastError());
        }

        /**
         * @brief Compute the forward pass of the layer.
         *
         * @param input The input tensor.
         * @param stream The CUDA stream used for launching the kernels.
         * @return The output tensor.
         */
        const core::Device_Tensor<data_type> &forward(
            const core::Device_Tensor<data_type> &input,
            const cudaStream_t stream = core::default_stream) override {
            // copy input for backward pass
            this->input.copy_from(input, stream);

            // preparation
            output.rebatchsize(input.batchsize(), stream);

            // compute forward pass
            if (use_bias) {
                dense_forward_copy_bias<data_type>
                    <<<device.blocks_2D(output.size() / out_features,
                                        out_features),
                       device.threads_2D(), 0, stream>>>(bias, output);
                core::checkCuda(cudaPeekAtLastError());
            }

            core::matrix_multiplication<data_type>(
                device.cublas_handle(stream), 1.0, input, in_features,
                CUBLAS_OP_N, weights, out_features, CUBLAS_OP_N, use_bias,
                output, output.size() / out_features, in_features,
                out_features);
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
         * @return The gradient with respect to the input.
         */
        const core::Device_Tensor<data_type> &backward(
            const core::Device_Tensor<data_type> &error,
            const cudaStream_t stream = core::default_stream) override {
            // preparation
            input_gradient.rebatchsize(error.batchsize(), stream);

            // calculate the gradient in respect to the input
            core::matrix_multiplication<data_type>(
                device.cublas_handle(stream), 1.0, error, out_features,
                CUBLAS_OP_N, weights, out_features, CUBLAS_OP_T, 0.0,
                input_gradient, input_gradient.size() / in_features,
                out_features, in_features);

            // calculate the gradient in respect to the weights
            core::matrix_multiplication<data_type>(
                device.cublas_handle(stream), 1.0, input, in_features,
                CUBLAS_OP_T, error, out_features, CUBLAS_OP_N, 0.0,
                weights_gradient, in_features, input.size() / in_features,
                out_features);

            // calculate the gradient in respect to the bias
            if (use_bias) {
                core::index_type n = error.size() / out_features;
                if (one.batchsize() != n) {
                    one.rebatchsize(n, stream);
                    core::set_one<data_type>
                        <<<device.blocks(one.size()), device.threads(), 0,
                           stream>>>(one);
                    core::checkCuda(cudaPeekAtLastError());
                }

                core::matrix_vector_multiplication<data_type>(
                    device.cublas_handle(stream), 1.0, error, n, out_features,
                    CUBLAS_OP_T, one, 0.0, bias_gradient);
                core::checkCuda(cudaPeekAtLastError());
            }

            // parameter updates
            if (training) {
                weights_optimizer.update(weights, weights_gradient, stream);

                if (use_bias)
                    bias_optimizer.update(bias, bias_gradient, stream);
            }

            return input_gradient;
        }

        /**
         * @brief Get the output of the forward pass.
         */
        const core::Device_Tensor<data_type> &get_output() const override {
            return output;
        }

        /**
         * @brief Get the gradient in respect to the input.
         */
        const core::Device_Tensor<data_type> &get_input_gradient()
            const override {
            return input_gradient;
        }

        /**
         * @brief Get the weights.
         */
        const core::Device_Tensor<data_type> &get_weights() const {
            return weights;
        }

        /**
         * @brief Get the bias.
         */
        const core::Device_Tensor<data_type> &get_bias() const {
            if (use_bias)
                return bias;
            else
                return null_tensor;
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
        core::index_type parameters() const override {
            core::index_type parameters = 0;
            parameters += weights.size();
            parameters += bias.size();
            return parameters;
        }

        /**
         * @brief Returns the memory footprint of the object on GPU.
         */
        const core::Mem_Size mem_size(
            const bool /* unused */ = false) const override {
            core::Mem_Size size;
            size += weights.mem_size(true);
            size += weights_gradient.mem_size(true);
            size += weights_optimizer.mem_size();
            size += bias.mem_size(true);
            size += bias_gradient.mem_size(true);
            size += bias_optimizer.mem_size();
            size += input.mem_size();
            size += output.mem_size();
            size += input_gradient.mem_size();
            size += null_tensor.mem_size(true);

            // one:
            size.fixed_size += one.mem_size().fixed_size;
            size.variable_size +=
                output.sample_size() / out_features * sizeof(data_type);

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
            if (use_bias) bias.save(path + "_bias", filetype);
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
            if (use_bias) bias.load(path + "_bias", filetype, stream);
        }
    };

}  // namespace layers
