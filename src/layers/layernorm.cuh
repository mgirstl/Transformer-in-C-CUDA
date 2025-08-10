/**
 * @file layernorm.cuh
 * @brief Implements the LayerNorm class.
 *
 * This file is part of the `layers` namespace, which implements differentiable
 * layers.
 */

#pragma once

#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
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
 * This specific file defines the LayerNorm class.
 *
 * @endinternal
 */
namespace layers {

    /**
     * @brief This kernel centers the input and squares it.
     *
     * @param input The input tensor containing the values to be processed. It
     * has to have the shape @f$ [*, *'] @f$.
     * @param input_centered The tensor to store the centered input in. It has
     * to have the shape @f$ [*, *'] @f$.
     * @param square The tensor to store the squared centered input in. It has
     * to have the shape @f$ [*, *'] @f$.
     * @param mean The mean along @f$ [*'] @f$. for all samples in @f$ [*] @f$.
     * It has to have the shape @f$ [*] @f$.
     *
     * @note The @f$ * @f$ symbol denotes any number of dimensions and must be
     * the same in all tensors where it is used.
     */
    template <core::arithmetic_type data_type>
    __global__ void layernorm_forward_square(
        const core::Kernel_Tensor<data_type> input,
        core::Kernel_Tensor<data_type> input_centered,
        core::Kernel_Tensor<data_type> square,
        const core::Kernel_Tensor<data_type> mean) {
        core::index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
        const core::index_type idx_stride = gridDim.x * blockDim.x;

        const core::index_type sample_size = input.size() / mean.size();
        for (; idx < input.size(); idx += idx_stride) {
            const data_type x = input[idx] - mean[idx / sample_size];
            input_centered[idx] = x;
            square[idx] = x * x;
        }
    }

    /**
     * @brief This kernel calculates from the variance the inverse standard
     * deviation.
     *
     * @param stat The tensor which holds the variance. For each element the
     * inverse is calculated and the root is drawn so that it is converted to
     * the inverse standard deviation. This transformation is made in-place.
     */
    template <core::arithmetic_type data_type>
    __global__ void layernorm_forward_std_inv(
        core::Kernel_Tensor<data_type> stat) {
        core::index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
        core::index_type idx_stride = gridDim.x * blockDim.x;

        constexpr data_type epsilon =
            std::numeric_limits<data_type>::epsilon() *
            std::numeric_limits<data_type>::epsilon();

        for (; idx < stat.size(); idx += idx_stride) {
            stat[idx] = sqrt(1. / (stat[idx] + epsilon));
        }
    }

    /**
     * @brief This kernel normalizes the input and then rescales and shifts it
     * with learned parameters.
     *
     * @param input_normalized This tensor first contains the centered input. In
     * this kernel this input will get normalized and again saved in this
     * tensor. It has to have the shape @f$ [*, *'] @f$.
     * @param output The tensor to store the end result in, i.e., this tensor
     * will contain the transformed normalized input. It has to have the shape
     * @f$ [*, *'] @f$.
     * @param std_inv The tensor that contains the inverse standard deviation
     * along @f$ [*'] @f$ for each sample in @f$ [*] @f$. It has to have the
     * shape @f$ [*] @f$.
     * @param gamma The learned scaling factors. It has to have the shape
     * @f$ [*'] @f$.
     * @param beta The learned shift. It has to have the shape @f$ [*'] @f$.
     *
     * @note The @f$ * @f$ symbol denotes any number of dimensions and must be
     * the same in all tensors where it is used.
     */
    template <core::arithmetic_type data_type>
    __global__ void layernorm_forward_transform(
        core::Kernel_Tensor<data_type> input_normalized,
        core::Kernel_Tensor<data_type> output,
        const core::Kernel_Tensor<data_type> std_inv,
        const core::Kernel_Tensor<data_type> gamma,
        const core::Kernel_Tensor<data_type> beta) {
        core::index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
        const core::index_type idx_stride = gridDim.x * blockDim.x;

        const core::index_type sample_size = gamma.size();
        for (; idx < input_normalized.size(); idx += idx_stride) {
            const data_type x =
                input_normalized[idx] * std_inv[idx / sample_size];
            output[idx] =
                gamma[idx % sample_size] * x + beta[idx % sample_size];
            input_normalized[idx] = x;
        }
    }

    /**
     * @brief This kernel adds up the different terms which constitute the loss
     * gradient in respect to the input of this layer.
     *
     * @param input_normalized This tensor first contains the normalized input
     * of the forward pass. It has to have the shape @f$ [*, *'] @f$.
     * @param error The input tensor containing the error values from the next
     * layer. It has to have the shape @f$ [*, *'] @f$.
     * @param gamma The learned scaling factors. It has to have the shape
     * @f$ [*'] @f$.
     * @param std_inv The tensor that contains the inverse standard deviation
     * along @f$ [*'] @f$ for each sample in @f$ [*] @f$. It has to have the
     * shape @f$ [*] @f$.
     * @param input_gradient The output tensor where the gradient values will be
     * stored in. It has to have the shape @f$ [*, *'] @f$.
     * @param error_sum The sum along the @f$ [*'] @f$ axes of the `error`. It
     * has to have the shape @f$ [*] @f$.
     * @param input_error_sum The sum along the @f$ [*] @f$ axes of the
     * element-wise multiplied `input_normalized` with the `error`. It has to
     * have the shape @f$ [*] @f$.
     *
     * @note The @f$ * @f$ symbol denotes any number of dimensions and must be
     * the same in all tensors where it is used.
     */
    template <core::arithmetic_type data_type>
    __global__ void layernorm_backward_input_gradient(
        const core::Kernel_Tensor<data_type> input_normalized,
        const core::Kernel_Tensor<data_type> error,
        const core::Kernel_Tensor<data_type> gamma,
        const core::Kernel_Tensor<data_type> std_inv,
        core::Kernel_Tensor<data_type> input_gradient,
        const core::Kernel_Tensor<data_type> error_sum,
        const core::Kernel_Tensor<data_type> input_error_sum) {
        core::index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
        const core::index_type idx_stride = gridDim.x * blockDim.x;

        const core::index_type sample_size = gamma.size();

        for (; idx < error.size(); idx += idx_stride) {
            const core::index_type row = idx / sample_size;
            const core::index_type col = idx % sample_size;

            const data_type term_1 = sample_size * gamma[col] * error[idx];
            const data_type term_2 = error_sum[row];
            const data_type term_3 =
                input_error_sum[row] * input_normalized[idx];

            input_gradient[idx] =
                std_inv[row] / sample_size * (term_1 - term_2 - term_3);
        }
    }

    /**
     * @class LayerNorm
     * @brief The LayerNorm layer.
     *
     * The LayerNorm layer computes:
     * @f[
     *     f(\vec{x}) = \vec{\gamma} \odot
     *                  \frac{\vec{x} - \mathbb{E}_i[x_i]}
     *                       {\sqrt{\mathbb{E}_i[x_i^2]+\epsilon}}
     *                + \vec{b}
     * @f]
     * for each sample @f$ \vec{x} @f$. Here @f$ \odot @f$ represents the
     * Hadamard product, @f$ \mathbb{E}_i[x_i] @f$ the mean of the elements of
     * @f$ \vec{x} @f$, and @f$ \mathbb{E}_i[x_i^2] @f$ the biased variance
     * estimate for those elements. @f$ \vec{\gamma} @f$ and @f$ \vec{\beta} @f$
     * are a learnable affine transform and @f$ \epsilon @f$ a small number.
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
    class LayerNorm final : public Layer<data_type, data_type, optimizer_type> {
      private:
        core::Device &device;
        core::index_type sample_size;
        core::Device_Tensor<data_type> stat;  // multipurpose variable
        core::Device_Tensor<data_type> gamma;
        core::Device_Tensor<data_type> gamma_gradient;
        optimizer_type<data_type> gamma_optimizer;
        core::Device_Tensor<data_type> beta;
        core::Device_Tensor<data_type> beta_gradient;
        optimizer_type<data_type> beta_optimizer;
        core::Device_Tensor<data_type> input_normalized;
        core::Device_Tensor<data_type> output;
        core::Device_Tensor<data_type> square;
        core::Device_Tensor<data_type> input_gradient;
        core::Device_Tensor<data_type> input_error_matrix;
        core::Device_Tensor<data_type> input_error_sum;
        core::Device_Tensor<data_type> error_sum;
        core::Device_Tensor<data_type> one;
        bool training;

      public:
        /**
         * @brief Construct a new LayerNorm object.
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param input_shape The shape (excluding the first axis) of the input
         * tensors.
         * @param normalized_shape The shape of the dimensions which get
         * normalized. If `input_shape` is @f$ [i_1, i_2, ..., i_n] @f$, then
         * the `normalized_shape` must satisfy one of the following conditions:
         * Either `normalized_shape` is a subset of `input_shape` starting from
         * any index @f$ j @f$ to @f$ n @f$. Specifically, `normalized_shape`
         * should be @f$ [i_j, i_{j+1}, ..., i_n] @f$, where
         * @f$ 1 \leq j \leq n @f$. Or  `normalized_shape` is the product of a
         * subset of input_shape starting from any index @f$ j @f$ to @f$ n @f$.
         * Specifically, `normalized_shape` should be
         * @f$ [i_j \cdot i_{j+1} \cdot ... \cdot i_n] @f$, where
         * @f$ 1 \leq j \leq n @f$.
         * @param optimizer_kw Will be passed to the optimizer. See the
         * documentation of the used optimizer to see which parameters are
         * possible.
         * @param stream The CUDA stream used for launching the kernels.
         *
         * @note Internally, the data elements described by the
         * `normalized_shape` are often called the samples.
         */
        LayerNorm(
            core::Device &device,
            const std::vector<core::index_type> &input_shape,
            const std::vector<core::index_type> &normalized_shape,
            const std::unordered_map<std::string, data_type> &optimizer_kw = {},
            const cudaStream_t stream = core::default_stream)
            : device{device},
              sample_size{std::accumulate(
                  normalized_shape.begin(), normalized_shape.end(),
                  core::index_type(1), std::multiplies<core::index_type>())},
              stat{core::NO_ALLOC, {}},
              gamma{normalized_shape},
              gamma_gradient{normalized_shape},
              gamma_optimizer{device, normalized_shape, optimizer_kw, stream},
              beta{normalized_shape},
              beta_gradient{normalized_shape},
              beta_optimizer{device, normalized_shape, optimizer_kw, stream},
              input_normalized{core::NO_ALLOC, input_shape},
              output{core::NO_ALLOC, input_shape},
              square{core::NO_ALLOC, input_shape},
              input_gradient{core::NO_ALLOC, input_shape},
              input_error_matrix{core::NO_ALLOC, input_shape},
              input_error_sum{core::NO_ALLOC, {}},
              error_sum{core::NO_ALLOC, {}},
              one{core::NO_ALLOC, {}},
              training{true} {
            core::set_one<data_type>
                <<<device.blocks(gamma.size()), device.threads(), 0, stream>>>(
                    gamma);
            core::checkCuda(cudaPeekAtLastError());

            core::checkCuda(cudaMemsetAsync(
                beta.data(), 0, beta.size() * sizeof(data_type), stream));
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
            // preparation
            input_normalized.rebatchsize(input.batchsize(), stream);
            square.rebatchsize(input.batchsize(), stream);
            output.rebatchsize(input.batchsize(), stream);

            core::index_type n = input.size() / sample_size;
            stat.rebatchsize(input.size() / sample_size, stream);

            // the batchsize of n is needed for the backward pass and the
            // batchsize of sample_size for the forward pass
            if (one.batchsize() != max(n, sample_size)) {
                one.rebatchsize(max(n, sample_size), stream);
                core::set_one<data_type><<<device.blocks(one.size()),
                                           device.threads(), 0, stream>>>(one);
                core::checkCuda(cudaPeekAtLastError());
            }

            // mean calculation
            core::matrix_vector_multiplication<data_type>(
                device.cublas_handle(stream), 1. / sample_size, input, n,
                sample_size, CUBLAS_OP_N, one, 0.0, stat);
            core::checkCuda(cudaPeekAtLastError());

            // variance calculation
            layernorm_forward_square<data_type>
                <<<device.blocks(input.size()), device.threads(), 0, stream>>>(
                    input, input_normalized, square, stat);
            core::checkCuda(cudaPeekAtLastError());

            core::matrix_vector_multiplication<data_type>(
                device.cublas_handle(stream), 1. / sample_size, square, n,
                sample_size, CUBLAS_OP_N, one, 0.0, stat);

            // inverse standard deviation calculation
            layernorm_forward_std_inv<data_type>
                <<<device.blocks(stat.size()), device.threads(), 0, stream>>>(
                    stat);
            core::checkCuda(cudaPeekAtLastError());

            // final transformation
            layernorm_forward_transform<data_type>
                <<<device.blocks(input.size()), device.threads(), 0, stream>>>(
                    input_normalized, output, stat, gamma, beta);
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
            input_error_matrix.rebatchsize(error.batchsize(), stream);

            core::index_type n = error.size() / sample_size;
            error_sum.rebatchsize(n, stream);
            input_error_sum.rebatchsize(n, stream);

            // calculate the gradient in respect to the input
            core::multiply<data_type>
                <<<device.blocks(error.size()), device.threads(), 0, stream>>>(
                    input_normalized, error, input_error_matrix);
            core::checkCuda(cudaPeekAtLastError());

            core::matrix_vector_multiplication<data_type>(
                device.cublas_handle(stream), 1.0, error, n, sample_size,
                CUBLAS_OP_N, gamma, 0.0, error_sum);

            core::matrix_vector_multiplication<data_type>(
                device.cublas_handle(stream), 1.0, input_error_matrix, n,
                sample_size, CUBLAS_OP_N, gamma, 0.0, input_error_sum);

            layernorm_backward_input_gradient<data_type>
                <<<device.blocks(error.size()), device.threads(), 0, stream>>>(
                    input_normalized, error, gamma, stat, input_gradient,
                    error_sum, input_error_sum);
            core::checkCuda(cudaPeekAtLastError());

            // calculate the gradient in respect to gamma
            core::matrix_vector_multiplication<data_type>(
                device.cublas_handle(stream), 1.0, input_error_matrix, n,
                sample_size, CUBLAS_OP_T, one, 0.0, gamma_gradient);

            // calculate the gradient in respect to beta
            core::matrix_vector_multiplication<data_type>(
                device.cublas_handle(stream), 1.0, error, n, sample_size,
                CUBLAS_OP_T, one, 0.0, beta_gradient);

            // parameter updates
            if (training) {
                gamma_optimizer.update(gamma, gamma_gradient, stream);
                beta_optimizer.update(beta, beta_gradient, stream);
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
         * @brief Get the parameters gamma.
         */
        const core::Device_Tensor<data_type> &get_gamma() const {
            return gamma;
        }

        /**
         * @brief Get the parameters beta.
         */
        const core::Device_Tensor<data_type> &get_beta() const { return beta; }

        /**
         * @brief Get the normalized input.
         */
        const core::Device_Tensor<data_type> &get_input_normalized() const {
            return input_normalized;
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
            parameters += gamma.size();
            parameters += beta.size();
            return parameters;
        }

        /**
         * @brief Returns the memory footprint of the object on GPU.
         */
        const core::Mem_Size mem_size(
            const bool /* unused */ = false) const override {
            core::Mem_Size size;
            size += gamma.mem_size(true);
            size += gamma_gradient.mem_size(true);
            size += gamma_optimizer.mem_size();
            size += beta.mem_size(true);
            size += beta_gradient.mem_size(true);
            size += beta_optimizer.mem_size();
            size += input_normalized.mem_size();
            size += output.mem_size();
            size += square.mem_size();
            size += input_gradient.mem_size();
            size += input_error_matrix.mem_size();

            // stat, input_error_sum and error_sum
            size.fixed_size += stat.mem_size().fixed_size;
            size.fixed_size += input_error_sum.mem_size().fixed_size;
            size.fixed_size += error_sum.mem_size().fixed_size;
            size.variable_size +=
                3 * output.sample_size() / sample_size * sizeof(data_type);

            // one has the size: max(batchsize * output.sample_size() /
            // sample_size, sample_size)
            size.fixed_size += one.mem_size().fixed_size;
            size.fixed_size += sample_size * sizeof(data_type);
            size.variable_size +=
                output.sample_size() / sample_size * sizeof(data_type);

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
            gamma.save(path + "_gamma", filetype);
            beta.save(path + "_beta", filetype);
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
            gamma.load(path + "_gamma", filetype, stream);
            beta.load(path + "_beta", filetype, stream);
        }
    };

}  // namespace layers
