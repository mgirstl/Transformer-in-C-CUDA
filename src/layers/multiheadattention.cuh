/**
 * @file multiheadattention.cuh
 * @brief Implements the MultiheadAttention class.
 *
 * This file is part of the `layers` namespace, which implements differentiable
 * layers.
 */

#pragma once

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "../activations/softmax.cuh"
#include "../core.cuh"
#include "dense.cuh"
#include "dropout.cuh"
#include "layer.cuh"

/**
 * @internal
 *
 * @namespace layers
 * @brief Implements differentiable layers.
 *
 * This specific file defines the MultiheadAttention class.
 *
 * @endinternal
 */
namespace layers {

    /**
     * @brief This kernel splits the input into several heads.
     *
     * The operation performed here corresponds to first reshaping the input
     * from @f$ [K, L, N \cdot M] @f$ to @f$ [K, L, N, M] @f$. And then
     * afterwards permuting the second and third axis.
     *
     * @param input The input tensor containing the values to be processed. It
     * has to have the shape @f$ [K, L, N \cdot M] @f$.
     * @param output The output tensor where the results will be stored in. It
     * has to have the shape @f$ [K, N, L, M] @f$.
     */
    template <core::float_type data_type>
    __global__ void multiheadattention_split_heads(
        const core::Kernel_Tensor<data_type> input,
        core::Kernel_Tensor<data_type> output) {
        core::index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
        const core::index_type stride = gridDim.x * blockDim.x;

        const core::index_type shape_1 = output.shape(1);
        const core::index_type shape_2 = output.shape(2);
        const core::index_type shape_3 = output.shape(3);

        const core::index_type input_area_z = shape_3;
        const core::index_type input_area_y = shape_1 * input_area_z;
        const core::index_type input_area_x = shape_2 * input_area_y;

        const core::index_type output_area_z = shape_3;
        const core::index_type output_area_y = shape_2 * output_area_z;
        const core::index_type output_area_x = shape_1 * output_area_y;

        for (; idx < output.size(); idx += stride) {
            core::index_type rest = idx;
            const core::index_type x = rest / output_area_x;

            rest = rest % output_area_x;
            const core::index_type y = rest / output_area_y;

            rest = rest % output_area_y;
            const core::index_type z = rest / output_area_z;
            const core::index_type w = rest % output_area_z;

            output[idx] = input[x * input_area_x + z * input_area_y +
                                y * input_area_z + w];
        }
    }

    /**
     * @brief This kernel groups the input from several heads into one.
     *
     * The operation performed here corresponds to first permuting the second
     * and third axis of the input and then reshaping it from
     * @f$ [K, L, N, M] @f$ to the shape @f$ [K, L, N \cdot M] @f$.
     *
     * @param input The input tensor containing the values to be processed. It
     * has to have the shape @f$ [K, N, L, M] @f$.
     * @param output The output tensor where the results will be stored in. It
     * has to have the shape @f$ [K, L, N \cdot M] @f$.
     */
    template <core::float_type data_type>
    __global__ void multiheadattention_group_heads(
        const core::Kernel_Tensor<data_type> input,
        core::Kernel_Tensor<data_type> output) {
        core::index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
        const core::index_type stride = gridDim.x * blockDim.x;

        const core::index_type shape_1 = input.shape(1);
        const core::index_type shape_2 = input.shape(2);
        const core::index_type shape_3 = input.shape(3);

        const core::index_type input_area_z = shape_3;
        const core::index_type input_area_y = shape_2 * input_area_z;
        const core::index_type input_area_x = shape_1 * input_area_y;

        const core::index_type output_area_z = shape_3;
        const core::index_type output_area_y = shape_1 * output_area_z;
        const core::index_type output_area_x = shape_2 * output_area_y;

        for (; idx < output.size(); idx += stride) {
            core::index_type rest = idx;
            const core::index_type x = rest / output_area_x;

            rest = rest % output_area_x;
            const core::index_type y = rest / output_area_y;

            rest = rest % output_area_y;
            const core::index_type z = rest / output_area_z;
            const core::index_type w = rest % output_area_z;

            output[idx] = input[x * input_area_x + z * input_area_y +
                                y * input_area_z + w];
        }
    }

    /**
     * @brief This kernel masks and scales the input tensor.
     *
     * For each element where the corresponding mask entry is 0, the input will
     * be set to a fixed value. For every other entry, the input will be scaled
     * with the scaling factor.
     *
     * @param tensor The tensor to apply the mask to. It must have the shape
     * @f$ [K, L, N, M] @f$.
     * @param mask The mask used for masking. It must have the shape
     * @f$ [K, M] @f$.
     * @param scale A scaling factor to scale the tensor elements.
     * @param value The value to set for the masked-out elements.
     *
     * @note The same mask will be used for all elements along the second and
     * third dimension (the @f$ L @f$ and @f$ N @f$ dimension).
     */
    template <core::float_type data_type>
    __global__ void multiheadattention_apply_mask_2D(
        core::Kernel_Tensor<data_type> tensor,
        const core::Kernel_Tensor<core::mask_type> mask, const data_type scale,
        const data_type value) {
        core::index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
        const core::index_type stride = gridDim.x * blockDim.x;

        const core::index_type area_z = tensor.shape(3);
        const core::index_type area_y = tensor.shape(2) * area_z;
        const core::index_type area_x = tensor.shape(1) * area_y;

        for (; idx < tensor.size(); idx += stride) {
            core::index_type rest = idx;
            const core::index_type x = rest / area_x;
            rest = rest % area_x;
            rest = rest % area_y;
            rest = rest % area_z;

            if (mask[x * mask.sample_size() + rest])
                tensor[idx] *= scale;
            else
                tensor[idx] = value;
        }
    }

    /**
     * @brief This kernel masks and scales the input tensor.
     *
     * For each element where the corresponding mask entry is 0, the input will
     * be set to a fixed value. For every other entry, the input will be scaled
     * with the scaling factor.
     *
     * @param tensor The tensor to apply the mask to. It must have the shape
     * @f$ [K, L, N, M] @f$.
     * @param mask The mask used for masking. It must have the shape
     * @f$ [K, N, M] @f$.
     * @param scale A scaling factor to scale the tensor elements.
     * @param value The value to set for the masked-out elements.
     *
     * @note The same mask will be used for all elements along the second
     * dimension (the @f$ L @f$ dimension).
     */
    template <core::float_type data_type>
    __global__ void multiheadattention_apply_mask_3D(
        core::Kernel_Tensor<data_type> tensor,
        const core::Kernel_Tensor<core::mask_type> mask, const data_type scale,
        const data_type value) {
        core::index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
        const core::index_type stride = gridDim.x * blockDim.x;

        const core::index_type area_y = tensor.shape(2) * tensor.shape(3);
        const core::index_type area_x = tensor.shape(1) * area_y;

        for (; idx < tensor.size(); idx += stride) {
            core::index_type rest = idx;
            const core::index_type x = rest / area_x;
            rest = rest % area_x;
            rest = rest % area_y;

            if (mask[x * mask.sample_size() + rest])
                tensor[idx] *= scale;
            else
                tensor[idx] = value;
        }
    }

    /**
     * @class MultiheadAttention
     * @brief The Multi-Head Attention layer.
     *
     * The Multi-Head Attention layer allows the model to jointly attend to
     * information from different representation subspaces at different
     * positions. This is a key component of the Transformer architecture.
     *
     * For more information, see "Attention Is All You Need".
     *
     * @note This layer is in training mode after initialization. This means
     * in this particular case that in the forward pass dropout is applied, and
     * in the backward pass the trainable parameters will be updated. In
     * contrast, in evaluation mode the dropout behaves like the identity
     * function and the parameter update is omitted.
     *
     * @tparam data_type The data type of the tensor elements.
     * @tparam optimizer_type The optimizer used for the parameter updates.
     */
    template <core::float_type data_type,
              template <typename> class optimizer_type>
    class MultiheadAttention final
        : public Layer<data_type, data_type, optimizer_type> {
      private:
        core::Device &device;
        core::index_type num_heads;
        core::index_type embedding_dim;
        core::index_type embeddings_per_head;
        core::index_type embedding_dim_prime;
        core::index_type query_length;
        core::index_type key_value_length;
        data_type scale;
        Dense<data_type, optimizer_type> q_linear;
        Dense<data_type, optimizer_type> k_linear;
        Dense<data_type, optimizer_type> v_linear;
        activations::Softmax<data_type> softmax;
        Dropout<data_type> dropout;
        Dense<data_type, optimizer_type> o_linear;
        core::Device_Tensor<data_type> query_split;
        core::Device_Tensor<data_type> key_split;
        core::Device_Tensor<data_type> value_split;
        core::Device_Tensor<data_type> output_split;
        core::Device_Tensor<data_type> output;
        core::Device_Tensor<data_type> energy;
        core::Device_Tensor<core::mask_type> mask;
        core::Device_Tensor<data_type> output_gradient_split;
        core::Device_Tensor<data_type> energy_gradient;
        core::Device_Tensor<data_type> query_gradient;
        core::Device_Tensor<data_type> key_gradient;
        core::Device_Tensor<data_type> value_gradient;
        core::Device_Tensor<data_type> query_gradient_split;
        core::Device_Tensor<data_type> key_gradient_split;
        core::Device_Tensor<data_type> value_gradient_split;
        core::Device_Tensor<data_type> null_tensor;
        bool training;

      public:
        /**
         * @brief Construct a new MultiheadAttention object.
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param query_shape The shape (excluding the first axis) of the query
         * tensors, i.e., `[query_length, embedding_dim]`.
         * @param key_value_shape The shape (excluding the first axis) of the
         * key and value tensors, i.e., `[key_value_length, embedding_dim]`.
         * @param optimizer_kw Will be passed to the optimizer. See the
         * documentation of the used optimizer to see which parameters are
         * possible.
         * @param num_heads Number of parallel attention heads.
         * @param dropout The probability that an element in the dropout layer
         * is set to zero.
         * @param stream The CUDA stream used for launching the kernels.
         */
        MultiheadAttention(
            core::Device &device,
            const std::vector<core::index_type> &query_shape,
            const std::vector<core::index_type> &key_value_shape,
            const std::unordered_map<std::string, data_type> &optimizer_kw = {},
            const core::index_type num_heads = 8, const data_type dropout = 0.1,
            const cudaStream_t stream = core::default_stream)
            : device{device},
              num_heads{num_heads},
              embedding_dim{query_shape.back()},
              embeddings_per_head{embedding_dim / num_heads},
              embedding_dim_prime{embeddings_per_head * num_heads},
              query_length{query_shape.front()},
              key_value_length{key_value_shape.front()},
              scale{1. / sqrt(embeddings_per_head)},
              q_linear{device,
                       query_shape,
                       {query_shape[0], embedding_dim_prime},
                       optimizer_kw,
                       false,
                       stream},
              k_linear{device,
                       key_value_shape,
                       {key_value_shape[0], embedding_dim_prime},
                       optimizer_kw,
                       false,
                       stream},
              v_linear{device,
                       key_value_shape,
                       {key_value_shape[0], embedding_dim_prime},
                       optimizer_kw,
                       false,
                       stream},
              softmax{device,
                      {num_heads, query_shape[0], key_value_shape[0]}},
              dropout{device,
                      {num_heads, query_shape[0], key_value_shape[0]},
                      dropout,
                      stream},
              o_linear{device,
                       {query_shape[0], embedding_dim_prime},
                       query_shape,
                       optimizer_kw,
                       true,
                       stream},
              query_split{core::NO_ALLOC,
                          {num_heads, query_length, embeddings_per_head}},
              key_split{core::NO_ALLOC,
                        {num_heads, key_value_length, embeddings_per_head}},
              value_split{core::NO_ALLOC,
                          {num_heads, key_value_length, embeddings_per_head}},
              output_split{core::NO_ALLOC,
                           {num_heads, query_length, embeddings_per_head}},
              output{core::NO_ALLOC, {query_length, embedding_dim_prime}},
              energy{core::NO_ALLOC,
                     {num_heads, query_length, key_value_length}},
              mask{core::NO_ALLOC, {query_length, key_value_length}},
              output_gradient_split{
                  core::NO_ALLOC,
                  {num_heads, query_length, embeddings_per_head}},
              energy_gradient{core::NO_ALLOC,
                              {num_heads, query_length, key_value_length}},
              query_gradient{core::NO_ALLOC,
                             {query_length, embedding_dim_prime}},
              key_gradient{core::NO_ALLOC,
                           {key_value_length, embedding_dim_prime}},
              value_gradient{core::NO_ALLOC,
                             {key_value_length, embedding_dim_prime}},
              query_gradient_split{
                  core::NO_ALLOC,
                  {num_heads, query_length, embeddings_per_head}},
              key_gradient_split{core::NO_ALLOC,
                                 {num_heads, key_value_length,
                                  embeddings_per_head, embedding_dim_prime}},
              value_gradient_split{
                  core::NO_ALLOC,
                  {num_heads, key_value_length, embeddings_per_head}},
              training{true} {}

#pragma diag_suppress virtual_function_decl_hidden

        /**
         * @brief Compute the forward pass of the layer.
         *
         * @note This function will also apply Dropout in training mode.
         *
         * @param query The input query. It has to have the shape
         * `[batchsize, query_length, embedding_dim]`.
         * @param key The input key. It has to have the shape
         * `[batchsize, key_value_length, embedding_dim]`.
         * @param value The input value. It has to have the shape
         * `[batchsize, key_value_length, embedding_dim]`.
         * @param mask The attention mask, i.e., the mask which specifies which
         * positions in the sequence get ignored in the attention calculation.
         * It has to have the shape `[batchsize, key_value_length]`
         * or `[batchsize, query_length, key_value_length]`.
         * @param stream The CUDA stream used for launching the kernels.
         * @return The output tensor.
         */
        const core::Device_Tensor<data_type> &forward(
            const core::Device_Tensor<data_type> &query,
            const core::Device_Tensor<data_type> &key,
            const core::Device_Tensor<data_type> &value,
            const core::Device_Tensor<core::mask_type> &mask,
            const cudaStream_t stream = core::default_stream) {
            // copy mask for backward pass
            this->mask.copy_from(mask, stream);

            // preparation
            core::index_type batchsize = query.batchsize();
            query_split.rebatchsize(batchsize, stream);
            key_split.rebatchsize(batchsize, stream);
            value_split.rebatchsize(batchsize, stream);

            energy.rebatchsize(batchsize, stream);
            output_split.rebatchsize(batchsize, stream);
            output.rebatchsize(batchsize, stream);

            // apply linear layers
            const auto &query_linear = q_linear.forward(query, stream);
            const auto &key_linear = k_linear.forward(key, stream);
            const auto &value_linear = v_linear.forward(value, stream);

            // split heads
            multiheadattention_split_heads<data_type>
                <<<device.blocks(query_split.size()), device.threads(), 0,
                   stream>>>(query_linear, query_split);
            core::checkCuda(cudaPeekAtLastError());

            multiheadattention_split_heads<data_type>
                <<<device.blocks(key_split.size()), device.threads(), 0,
                   stream>>>(key_linear, key_split);
            core::checkCuda(cudaPeekAtLastError());

            multiheadattention_split_heads<data_type>
                <<<device.blocks(value_split.size()), device.threads(), 0,
                   stream>>>(value_linear, value_split);
            core::checkCuda(cudaPeekAtLastError());

            // calculate energy
            core::strided_batched_matrix_multiplication<data_type>(
                device.cublas_handle(stream), scale, query_split,
                embeddings_per_head, query_length * embeddings_per_head,
                CUBLAS_OP_N, key_split, embeddings_per_head,
                key_value_length * embeddings_per_head, CUBLAS_OP_T, 0, energy,
                query_length, embeddings_per_head, key_value_length,
                query_length * key_value_length,
                energy.batchsize() * num_heads);

            // mask energy
            if (mask.sample_size() == key_value_length)
                multiheadattention_apply_mask_2D<data_type>
                    <<<device.blocks(energy.size()), device.threads(), 0,
                       stream>>>(energy, mask, 1, -INFINITY);
            else
                multiheadattention_apply_mask_3D<data_type>
                    <<<device.blocks(energy.size()), device.threads(), 0,
                       stream>>>(energy, mask, 1, -INFINITY);

            core::checkCuda(cudaPeekAtLastError());

            // calculate attention
            const auto &attention = softmax.forward(energy, stream);
            const auto &dropout_attention = dropout.forward(attention, stream);

            // calculate final output
            core::strided_batched_matrix_multiplication<data_type>(
                device.cublas_handle(stream), 1, dropout_attention,
                key_value_length, query_length * key_value_length, CUBLAS_OP_N,
                value_split, embeddings_per_head,
                key_value_length * embeddings_per_head, CUBLAS_OP_N, 0,
                output_split, query_length, key_value_length,
                embeddings_per_head, query_length * embeddings_per_head,
                output_split.batchsize() * num_heads);

            // group heads
            multiheadattention_group_heads<data_type>
                <<<device.blocks(output_split.size()), device.threads(), 0,
                   stream>>>(output_split, output);
            core::checkCuda(cudaPeekAtLastError());

            return o_linear.forward(output, stream);
        }

#pragma diag_default virtual_function_decl_hidden

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
         * `input_gradient`. The reason for this is that the forward pass has
         * multiple input tensors.
         */
        const core::Device_Tensor<data_type> &backward(
            const core::Device_Tensor<data_type> &error,
            const cudaStream_t stream = core::default_stream) override {
            // preparation
            core::index_type batchsize = error.batchsize();
            query_gradient.rebatchsize(batchsize, stream);
            query_gradient_split.rebatchsize(batchsize, stream);
            key_gradient.rebatchsize(batchsize, stream);
            key_gradient_split.rebatchsize(batchsize, stream);
            value_gradient.rebatchsize(batchsize, stream);
            value_gradient_split.rebatchsize(batchsize, stream);
            output_gradient_split.rebatchsize(batchsize, stream);
            energy_gradient.rebatchsize(batchsize, stream);

            // calculate the gradient in respect to the output
            const auto &output_gradient = o_linear.backward(error, stream);

            multiheadattention_split_heads<data_type>
                <<<device.blocks(output_gradient_split.size()),
                   device.threads(), 0, stream>>>(output_gradient,
                                                  output_gradient_split);
            core::checkCuda(cudaPeekAtLastError());

            // calculate the gradient in respect to the energy
            core::strided_batched_matrix_multiplication<data_type>(
                device.cublas_handle(stream), 1, output_gradient_split,
                embeddings_per_head, query_length * embeddings_per_head,
                CUBLAS_OP_N, value_split, embeddings_per_head,
                key_value_length * embeddings_per_head, CUBLAS_OP_T, 0,
                energy_gradient, query_length, embeddings_per_head,
                key_value_length, query_length * key_value_length,
                energy_gradient.batchsize() * num_heads);

            const auto &attention_gradient =
                dropout.backward(energy_gradient, stream);
            const auto &temp = softmax.backward(attention_gradient, stream);
            energy_gradient.copy_from(temp, stream);

            if (mask.sample_size() == key_value_length)
                multiheadattention_apply_mask_2D<data_type>
                    <<<device.blocks(energy_gradient.size()), device.threads(),
                       0, stream>>>(energy_gradient, mask, scale, 0);
            else
                multiheadattention_apply_mask_3D<data_type>
                    <<<device.blocks(energy_gradient.size()), device.threads(),
                       0, stream>>>(energy_gradient, mask, scale, 0);

            core::checkCuda(cudaPeekAtLastError());

            // calculate the gradient in respect to the query
            strided_batched_matrix_multiplication<data_type>(
                device.cublas_handle(stream), 1, energy_gradient,
                key_value_length, query_length * key_value_length, CUBLAS_OP_N,
                key_split, embeddings_per_head,
                key_value_length * embeddings_per_head, CUBLAS_OP_N, 0,
                query_gradient_split, query_length, key_value_length,
                embeddings_per_head, query_length * embeddings_per_head,
                energy_gradient.batchsize() * num_heads);

            // calculate the gradient in respect to the key
            strided_batched_matrix_multiplication<data_type>(
                device.cublas_handle(stream), 1, energy_gradient,
                key_value_length, query_length * key_value_length, CUBLAS_OP_T,
                query_split, embeddings_per_head,
                query_length * embeddings_per_head, CUBLAS_OP_N, 0,
                key_gradient_split, key_value_length, query_length,
                embeddings_per_head, key_value_length * embeddings_per_head,
                energy_gradient.batchsize() * num_heads);

            // calculate the gradient in respect to the value
            core::strided_batched_matrix_multiplication<data_type>(
                device.cublas_handle(stream), 1, dropout.get_output(),
                key_value_length, query_length * key_value_length, CUBLAS_OP_T,
                output_gradient_split, embeddings_per_head,
                query_length * embeddings_per_head, CUBLAS_OP_N, 0,
                value_gradient_split, key_value_length, query_length,
                embeddings_per_head, key_value_length * embeddings_per_head,
                output_gradient_split.batchsize() * num_heads);

            // group heads
            multiheadattention_group_heads<data_type>
                <<<device.blocks(query_gradient_split.size()), device.threads(),
                   0, stream>>>(query_gradient_split, query_gradient);
            core::checkCuda(cudaPeekAtLastError());

            multiheadattention_group_heads<data_type>
                <<<device.blocks(key_gradient_split.size()), device.threads(),
                   0, stream>>>(key_gradient_split, key_gradient);
            core::checkCuda(cudaPeekAtLastError());

            multiheadattention_group_heads<data_type>
                <<<device.blocks(value_gradient_split.size()), device.threads(),
                   0, stream>>>(value_gradient_split, value_gradient);
            core::checkCuda(cudaPeekAtLastError());

            // linear backward
            q_linear.backward(query_gradient, stream);
            k_linear.backward(key_gradient, stream);
            v_linear.backward(value_gradient, stream);

            return null_tensor;
        }

        /**
         * @brief Get the output of the forward pass.
         */
        const core::Device_Tensor<data_type> &get_output() const override {
            return output;
        }

        /**
         * @brief Get the in the forward pass calculated attention.
         */
        const core::Device_Tensor<data_type> &get_attention() const {
            return softmax.get_output();
        }

        /**
         * @brief Get the gradient in respect to the query.
         */
        const core::Device_Tensor<data_type> &get_query_gradient() const {
            return q_linear.get_input_gradient();
        }

        /**
         * @brief Get the gradient in respect to the key.
         */
        const core::Device_Tensor<data_type> &get_key_gradient() const {
            return k_linear.get_input_gradient();
        }

        /**
         * @brief Get the gradient in respect to the value.
         */
        const core::Device_Tensor<data_type> &get_value_gradient() const {
            return v_linear.get_input_gradient();
        }

        /**
         * @brief Get the query linear layer.
         */
        const Dense<data_type, optimizer_type> &get_query_linear() const {
            return q_linear;
        }

        /**
         * @brief Get the key linear layer.
         */
        const Dense<data_type, optimizer_type> &get_key_linear() const {
            return k_linear;
        }

        /**
         * @brief Get the value linear layer.
         */
        const Dense<data_type, optimizer_type> &get_value_linear() const {
            return v_linear;
        }

        /**
         * @brief Get the softmax function.
         */
        const activations::Softmax<data_type> &get_softmax() const {
            return softmax;
        }

        /**
         * @brief Get the dropout layer.
         */
        const Dropout<data_type> &get_dropout() const { return dropout; }

        /**
         * @brief Get the output linear layer.
         */
        const Dense<data_type, optimizer_type> &get_output_linear() const {
            return o_linear;
        }

        /**
         * @brief Sets the layer into training mode.
         *
         * In training mode, dropout is applied in the forward pass, and in the
         * backward pass the trainable parameters will be updated.
         *
         * This layer is initially set to training mode.
         */
        void train() override {
            q_linear.train();
            k_linear.train();
            v_linear.train();
            dropout.train();
            o_linear.train();
        }

        /**
         * @brief Sets the layer into evaluation mode.
         *
         * In evaluation mode, the dropout behaves like the identity function
         * and the parameter update is omitted in the backward pass.
         *
         * This layer is initially set to training mode.
         */
        void eval() override {
            q_linear.eval();
            k_linear.eval();
            v_linear.eval();
            dropout.eval();
            o_linear.eval();
        }

        /**
         * @brief Returns the number of trainable parameters of the layer.
         */
        core::index_type parameters() const override {
            core::index_type parameters = 0;
            parameters += q_linear.parameters();
            parameters += k_linear.parameters();
            parameters += v_linear.parameters();
            parameters += o_linear.parameters();
            return parameters;
        }

        /**
         * @brief Returns the memory footprint of the object on GPU.
         */
        const core::Mem_Size mem_size(
            const bool /* unused */ = false) const override {
            core::Mem_Size size;
            size += q_linear.mem_size();
            size += k_linear.mem_size();
            size += v_linear.mem_size();
            size += softmax.mem_size();
            size += dropout.mem_size();
            size += o_linear.mem_size();
            size += query_split.mem_size();
            size += key_split.mem_size();
            size += value_split.mem_size();
            size += output_split.mem_size();
            size += output.mem_size();
            size += energy.mem_size();
            size += output_gradient_split.mem_size();
            size += energy_gradient.mem_size();
            size += query_gradient.mem_size();
            size += key_gradient.mem_size();
            size += value_gradient.mem_size();
            size += query_gradient_split.mem_size();
            size += key_gradient_split.mem_size();
            size += value_gradient_split.mem_size();
            size += null_tensor.mem_size(true);

            // mask has the size: max(batchsize * query_length *
            // key_value_length, batchsize * query_length * key_value_length)
            size.fixed_size += mask.mem_size().fixed_size;
            size.variable_size +=
                query_length * key_value_length * sizeof(data_type);

            return size;
        }

        /**
         * @brief Saves the parameters of the layer to a file.
         *
         * @param path The base path to write the parameters to.
         * @param filetype The filetype of the saved files.
         */
        void _save(const std::string &path,
                   const std::string &filetype) override {
            q_linear.save(path + "_q", filetype);
            k_linear.save(path + "_k", filetype);
            v_linear.save(path + "_v", filetype);
            o_linear.save(path + "_o", filetype);
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
            q_linear.load(path + "_q", stream);
            k_linear.load(path + "_k", stream);
            v_linear.load(path + "_v", stream);
            o_linear.load(path + "_o", stream);
        }
    };

}  // namespace layers
