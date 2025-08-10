/**
 * @file transformer.cuh
 * @brief Implements the Transformer class.
 *
 * This file is part of the `layers` namespace, which implements differentiable
 * layers.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "../core.cuh"
#include "decoder.cuh"
#include "encoder.cuh"

/**
 * @internal
 *
 * @namespace layers
 * @brief Implements differentiable layers.
 *
 * This specific file defines the Transformer class.
 *
 * @endinternal
 */
namespace layers {

    /**
     * @brief This kernel creates the attention mask for the source tensor of a
     * transformer model.
     *
     * The operation performed here involves checking each element of the source
     * tensor and setting the corresponding element in the source mask tensor to
     * indicate whether the element should be ignored or not.
     *
     * @param source The input tensor containing the source values to be
     * processed.
     * @param source_mask The output tensor where the source mask will be
     * stored in.
     * @param ignore_index The value in the source tensor that should be
     * ignored.
     *
     * @note It is assumed that all the tensors have the same size.
     */
    template <core::integer_type data_type>
    __global__ void transformer_pad_source(
        const core::Kernel_Tensor<data_type> source,
        core::Kernel_Tensor<core::mask_type> source_mask,
        const data_type ignore_index) {
        core::index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
        const core::index_type stride = gridDim.x * blockDim.x;

        for (; idx < source.size(); idx += stride)
            source_mask[idx] = source[idx] != ignore_index;
    }

    /**
     * @brief This kernel creates the attention mask for the target tensor of a
     * transformer model.
     *
     * The operation performed here involves checking each element of the target
     * tensor and setting the corresponding element in the target mask tensor to
     * indicate whether the element should be ignored or not. Additionally, it
     * ensures that future positions in the target sequence are masked.
     *
     * @param target The input tensor containing the target values to be
     * processed. It has to have the shape @f$ [N, M] @f$.
     * @param target_mask The output tensor where the target mask will be
     * stored in. It has to have the shape @f$ [N, M, M] @f$.
     * @param ignore_index The value in the target tensor that should be
     * ignored.
     */
    template <core::integer_type data_type>
    __global__ void transformer_pad_target(
        const core::Kernel_Tensor<data_type> target,
        core::Kernel_Tensor<core::mask_type> target_mask,
        const data_type ignore_index) {
        const core::index_type batch = blockIdx.x * blockDim.x + threadIdx.x;
        const core::index_type batch_stride = gridDim.x * blockDim.x;
        const core::index_type row = blockIdx.y * blockDim.y + threadIdx.y;
        const core::index_type row_stride = gridDim.y * blockDim.y;
        const core::index_type col = blockIdx.z * blockDim.z + threadIdx.z;
        const core::index_type col_stride = gridDim.z * blockDim.z;

        for (core::index_type b = batch; b < target_mask.shape(0);
             b += batch_stride) {
            for (core::index_type r = row; r < target_mask.shape(1);
                 r += row_stride) {
                for (core::index_type c = col; c < target_mask.shape(2);
                     c += col_stride) {
                    const core::index_type idx = b * target_mask.sample_size() +
                                                 r * target_mask.shape(2) + c;
                    if (c > r)
                        target_mask[idx] = 0;
                    else
                        target_mask[idx] =
                            target[b * target.sample_size() + c] !=
                            ignore_index;
                }
            }
        }
    }

    /**
     * @class Transformer
     * @brief The Transformer layer.
     *
     * The Transformer model consists of an Encoder and a Decoder. This
     * architecture is designed to handle sequential data and is widely used in
     * natural language processing tasks. The Transformer model leverages
     * self-attention mechanisms to process input sequences in parallel, making
     * it highly efficient.
     *
     * The output of the Transformer are the probabilities of each token in the
     * vocabulary being the next token in the sequence.
     *
     * For more information, see "Attention Is All You Need".
     *
     * @tparam input_type The data type of the input tensor elements, i.e., the
     * type of the source and target tensor.
     * @tparam output_type The data type of the output tensor elements.
     * @tparam optimizer_type The optimizer used for the parameter updates.
     */
    template <core::integer_type input_type, core::float_type output_type,
              template <typename> class optimizer_type>
    class Transformer final
        : public Layer<input_type, output_type, optimizer_type> {
      private:
        core::Device &device;
        input_type ignore_index;
        Encoder<input_type, output_type, optimizer_type> encoder;
        Decoder<input_type, output_type, optimizer_type> decoder;
        core::Device_Tensor<core::mask_type> source_mask;
        core::Device_Tensor<core::mask_type> target_mask;
        core::Device_Tensor<input_type> null_tensor;

      public:
        /**
         * @brief Construct a new Transformer object.
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param source_shape The shape (excluding the first axis) of the
         * Encoder input, i.e., `[source_sequence_length]`.
         * @param target_shape The shape (excluding the first axis) of the main
         * Decoder input, i.e., `[target_sequence_length]`.
         * @param embedding_dim The dimension of the embedding vectors.
         * @param hidden_dim The dimension of the features in the hidden layer
         * in the Positionwise Feed Forward layer.
         * @param num_encoder_layers The number of EncoderLayers.
         * @param num_decoder_layers The number of DecoderLayers.
         * @param num_source_embeddings The size of the source dictionary, i.e.,
         * this value corresponds to one larger than the largest value in the
         * source tensor.
         * @param num_target_embeddings The size of the target dictionary, i.e.,
         * this value corresponds to one larger than the largest value in the
         * target tensor.
         * @param optimizer_kw Will be passed to the optimizer. See the
         * documentation of the used optimizer to see which parameters are
         * possible.
         * @param num_heads Number of parallel attention heads.
         * @param dropout The probability that an element in the dropout layer
         * is set to zero.
         * @param ignore_index The value in the input tensors that should be
         * ignored.
         * @param max_batchsize The maximum batchsize to be expected.
         * @param stream The CUDA stream used for launching the kernels.
         *
         * @note The `output_shape` of the layer is automatically derived from
         * the other arguments. It will be
         * `[batchsize, target_sequence_length, num_target_embeddings]`.
         */
        Transformer(core::Device &device,
                    const std::vector<core::index_type> &source_shape,
                    const std::vector<core::index_type> &target_shape,
                    const core::index_type embedding_dim,
                    const core::index_type hidden_dim,
                    const core::index_type num_encoder_layers,
                    const core::index_type num_decoder_layers,
                    const input_type num_source_embeddings,
                    const input_type num_target_embeddings,
                    const std::unordered_map<std::string, output_type>
                        &optimizer_kw = {},
                    const core::index_type num_heads = 8,
                    const core::index_type dropout = 0.1,
                    const input_type ignore_index =
                        std::numeric_limits<input_type>::max(),
                    const core::index_type max_batchsize = 5000,
                    const cudaStream_t stream = core::default_stream)
            : device{device},
              ignore_index{ignore_index},
              encoder{device,
                      source_shape,
                      extend_shape(source_shape, embedding_dim),
                      extend_shape(source_shape, hidden_dim),
                      num_encoder_layers,
                      num_source_embeddings,
                      optimizer_kw,
                      num_heads,
                      dropout,
                      max_batchsize,
                      stream},
              decoder{device,
                      target_shape,
                      extend_shape(source_shape, embedding_dim),
                      extend_shape(target_shape, hidden_dim),
                      num_decoder_layers,
                      num_target_embeddings,
                      optimizer_kw,
                      num_heads,
                      dropout,
                      max_batchsize,
                      stream},
              source_mask{core::NO_ALLOC, {1, source_shape.back()}},
              target_mask{core::NO_ALLOC,
                          {target_shape.back(), target_shape.back()}} {}

#pragma diag_suppress virtual_function_decl_hidden

        /**
         * @brief Compute the forward pass of the layer.
         *
         * @note This function will also apply Dropout in training mode.
         *
         * @param source The input tensor for the Encoder. It has to have the
         * shape `[batchsize, source_sequence_length]`.
         * @param target The main input tensor for the Decoder. It has to have
         * the shape `[batchsize, target_sequence_length]`.
         * @param stream The CUDA stream used for launching the kernels.
         * @return The output tensor.
         */
        const core::Device_Tensor<output_type> &forward(
            const core::Device_Tensor<input_type> &source,
            const core::Device_Tensor<input_type> &target,
            const cudaStream_t stream = core::default_stream) {
            // preparation
            source_mask.rebatchsize(source.batchsize(), stream);
            target_mask.rebatchsize(target.batchsize(), stream);

            // calculate padding masks
            transformer_pad_source<input_type>
                <<<device.blocks(source.size()), device.threads(), 0, stream>>>(
                    source, source_mask, ignore_index);
            core::checkCuda(cudaPeekAtLastError());

            transformer_pad_target<input_type>
                <<<device.blocks_3D(target.batchsize(), target.sample_size(),
                                    target.sample_size()),
                   device.threads_3D(), 0, stream>>>(target, target_mask,
                                                     ignore_index);
            core::checkCuda(cudaPeekAtLastError());

            // launch encoder and decoder
            const auto &x = encoder.forward(source, source_mask, stream);
            return decoder.forward(target, target_mask, x, source_mask, stream);
        }

#pragma diag_suppress virtual_function_decl_hidden

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
         * multiple inputs.
         */
        const core::Device_Tensor<input_type> &backward(
            const core::Device_Tensor<output_type> &error,
            const cudaStream_t stream = core::default_stream) {
            decoder.backward(error, stream);
            const auto &x = decoder.get_source_gradient();
            encoder.backward(x, stream);
            return null_tensor;
        }

        /**
         * @brief Get the output of the forward pass.
         */
        const core::Device_Tensor<output_type> &get_output() const override {
            return decoder.get_output();
        }

        /**
         * @brief Get the in the forward pass calculated attention.
         */
        const core::Device_Tensor<output_type> &get_attention() const {
            return decoder.get_attention();
        }

        /**
         * @brief Get the gradient with respect to the source.
         *
         * @attention This function returns an empty tensor instead of the
         * `source_gradient`. The reason for this is that the `source_gradient`
         * is not defined for an integer source.
         */
        const core::Device_Tensor<input_type> &get_source_gradient() const {
            return null_tensor;
        }

        /**
         * @brief Get the gradient with respect to the target.
         *
         * @attention This function returns an empty tensor instead of the
         * `target_gradient`. The reason for this is that the `target_gradient`
         * is not defined for an integer target.
         */
        const core::Device_Tensor<input_type> &get_target_gradient() const {
            return null_tensor;
        }

        /**
         * @brief Get the source mask.
         */
        const core::Device_Tensor<core::mask_type> &get_source_mask() const {
            return source_mask;
        }

        /**
         * @brief Get the target mask.
         */
        const core::Device_Tensor<core::mask_type> &get_target_mask() const {
            return target_mask;
        }

        /**
         * @brief Get the Encoder layer.
         */
        const Encoder<input_type, output_type, optimizer_type> &get_encoder()
            const {
            return encoder;
        }

        /**
         * @brief Get the Decoder layer.
         */
        const Decoder<input_type, output_type, optimizer_type> &get_decoder()
            const {
            return decoder;
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
            encoder.train();
            decoder.train();
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
            encoder.eval();
            decoder.eval();
        }

        /**
         * @brief Returns the number of trainable parameters of the layer.
         */
        core::index_type parameters() const override {
            core::index_type parameters = 0;
            parameters += encoder.parameters();
            parameters += decoder.parameters();
            return parameters;
        }

        /**
         * @brief Returns the memory footprint of the object on GPU.
         */
        const core::Mem_Size mem_size(
            const bool /* unused */ = false) const override {
            core::Mem_Size size;
            size += encoder.mem_size();
            size += decoder.mem_size();
            size += source_mask.mem_size();
            size += target_mask.mem_size();
            size += null_tensor.mem_size(true);
            return size;
        }

      private:
        /**
         * @brief Extends a given shape vector with an additional value at the
         * end.
         */
        static std::vector<core::index_type> extend_shape(
            const std::vector<core::index_type> &shape,
            const core::index_type value) {
            auto temp{shape};
            temp.push_back(value);
            return temp;
        }

        /**
         * @brief Saves the parameters of the layer to a file.
         *
         * @param path The base path to write the parameters to.
         * @param filetype The filetype of the saved files.
         */
        void _save(const std::string &path,
                   const std::string &filetype) override {
            encoder.save(path + "_en", filetype);
            decoder.save(path + "_de", filetype);
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
            encoder.load(path + "_en", filetype, stream);
            decoder.load(path + "_de", filetype, stream);
        }
    };

}  // namespace layers
