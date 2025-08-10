/**
 * @file encoder.cuh
 * @brief Implements the Encoder class.
 *
 * This file is part of the `layers` namespace, which implements differentiable
 * layers.
 */

#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../core.cuh"
#include "embedding.cuh"
#include "encoderlayer.cuh"
#include "layer.cuh"
#include "positionalencoding.cuh"

/**
 * @internal
 *
 * @namespace layers
 * @brief Implements differentiable layers.
 *
 * This specific file defines the Encoder class.
 *
 * @endinternal
 */
namespace layers {

    /**
     * @class Encoder
     * @brief The Encoder layer.
     *
     * The Encoder consists of a stack of EncoderLayers. This is a key component
     * of the Transformer architecture.
     *
     * For more information, see "Attention Is All You Need".
     *
     * @tparam input_type The data type of the input tensor elements.
     * @tparam output_type The data type of the output tensor elements.
     * @tparam optimizer_type The optimizer used for the parameter updates.
     */
    template <core::integer_type input_type, core::float_type output_type,
              template <typename> class optimizer_type>
    class Encoder final
        : public Layer<input_type, output_type, optimizer_type> {
      private:
        Embedding<input_type, output_type, optimizer_type> embedding;
        PositionalEncoding<output_type> positionalencoding;
        std::vector<EncoderLayer<output_type, optimizer_type>> layers;
        core::Device_Tensor<input_type> null_tensor;

      public:
        /**
         * @brief Construct a new Encoder object.
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param input_shape The shape (excluding the first axis) of the input
         * tensors, i.e., `[sequence_length]`.
         * @param output_shape The shape (excluding the first axis) of the
         * output tensors, i.e., `[sequence_length, embedding_dim]`.
         * @param hidden_shape The shape (excluding the first axis) of the
         * features in the hidden dimension of the Positionwise Feed Forward
         * layer, i.e., `[sequence_length, hidden_dim]`.
         * @param num_layers The number of EncoderLayers.
         * @param num_embeddings The size of the dictionary, i.e., this value
         * corresponds to one larger than the largest value in the input tensor.
         * @param optimizer_kw Will be passed to the optimizer. See the
         * documentation of the used optimizer to see which parameters are
         * possible.
         * @param num_heads Number of parallel attention heads.
         * @param dropout The probability that an element in the dropout layer
         * is set to zero.
         * @param max_batchsize The maximum batchsize to be expected.
         * @param stream The CUDA stream used for launching the kernels.
         *
         * @throws std::invalid_argument if `num_layers` is smaller than 1.
         */
        Encoder(core::Device &device,
                const std::vector<core::index_type> &input_shape,
                const std::vector<core::index_type> &output_shape,
                const std::vector<core::index_type> &hidden_shape,
                const core::index_type num_layers,
                const input_type num_embeddings,
                const std::unordered_map<std::string, output_type>
                    &optimizer_kw = {},
                const core::index_type num_heads = 8,
                const output_type dropout = 0.1,
                const core::index_type max_batchsize = 5000,
                const cudaStream_t stream = core::default_stream)
            : embedding{device,
                        input_shape,
                        output_shape,
                        num_embeddings,
                        optimizer_kw,
                        sqrt(output_shape.back()),
                        stream},
              positionalencoding{device,
                                 output_shape,
                                 dropout,
                                 max_batchsize,
                                 stream} {
            if (num_layers < 1)
                throw std::invalid_argument(
                    "num_layers needs to be at least 1!");

            layers.reserve(num_layers);
            for (core::index_type idx = 0; idx < num_layers; ++idx)
                layers.emplace_back(device, output_shape, hidden_shape,
                                    optimizer_kw, num_heads, dropout, stream);
        }

#pragma diag_suppress virtual_function_decl_hidden

        /**
         * @brief Compute the forward pass of the layer.
         *
         * @note This function will also apply Dropout in training mode.
         *
         * @param input The input tensor. It has to have the shape
         * `[batchsize, sequence_length]`.
         * @param mask The attention mask, i.e., the mask which specifies which
         * positions in the sequence get ignored in the attention calculation.
         * It has to have the shape `[batchsize, sequence_length]` or
         * `[batchsize, sequence_length, sequence_length]`.
         * @param stream The CUDA stream used for launching the kernels.
         * @return The output tensor.
         */
        const core::Device_Tensor<output_type> &forward(
            const core::Device_Tensor<input_type> &input,
            const core::Device_Tensor<core::mask_type> &mask,
            const cudaStream_t stream = core::default_stream) {
            const auto &x1 = embedding.forward(input, stream);
            const auto &x2 = positionalencoding.forward(x1, stream);

            layers.front().forward(x2, mask, stream);

            for (core::index_type idx = 1; idx < layers.size(); ++idx) {
                const auto &x3 = layers[idx - 1].get_output();
                layers[idx].forward(x3, mask, stream);
            }

            return layers.back().get_output();
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
         * `input_gradient`. The reason for this is that the `input_gradient` is
         * not defined for an integer input.
         */
        const core::Device_Tensor<input_type> &backward(
            const core::Device_Tensor<output_type> &error,
            const cudaStream_t stream = core::default_stream) override {
            layers.back().backward(error, stream);

            for (core::index_type idx = 1; idx < layers.size(); ++idx) {
                const core::index_type _idx = layers.size() - idx - 1;
                const auto &x1 = layers[_idx + 1].get_input_gradient();
                layers[_idx].backward(x1, stream);
            }

            const auto &x2 = layers.front().get_input_gradient();
            const auto &x3 = positionalencoding.backward(x2, stream);
            embedding.backward(x3, stream);

            return null_tensor;
        }

        /**
         * @brief Get the output of the forward pass.
         */
        const core::Device_Tensor<output_type> &get_output() const override {
            return layers.back().get_output();
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
         * @brief Get the Embedding layer.
         */
        const Embedding<input_type, output_type, optimizer_type>
            &get_embedding() const {
            return embedding;
        }

        /**
         * @brief Get the PositionalEncoding layer.
         */
        const PositionalEncoding<output_type> &get_positionalencoding() const {
            return positionalencoding;
        }

        /**
         * @brief Get all the EncoderLayers as a std::vector.
         */
        const std::vector<EncoderLayer<output_type, optimizer_type>>
            &get_layers() const {
            return layers;
        }

        /**
         * @brief Get a specific EncoderLayer.
         */
        const EncoderLayer<output_type, optimizer_type> &get_layer(
            const core::index_type idx) const {
            return layers[idx];
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
            embedding.train();
            positionalencoding.train();
            for (auto &layer : layers) layer.train();
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
            embedding.eval();
            positionalencoding.eval();
            for (auto &layer : layers) layer.train();
        }

        /**
         * @brief Returns the number of trainable parameters of the layer.
         */
        core::index_type parameters() const override {
            core::index_type parameters = 0;
            parameters += embedding.parameters();
            for (auto &layer : layers) parameters += layer.parameters();
            return parameters;
        }

        /**
         * @brief Returns the memory footprint of the object on GPU.
         */
        const core::Mem_Size mem_size(
            const bool /* unused */ = false) const override {
            core::Mem_Size size;
            size += embedding.mem_size();
            size += positionalencoding.mem_size();
            for (auto &layer : layers) size += layer.mem_size();
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
            embedding.save(path + "_e", filetype);
            for (core::index_type idx = 0; idx < layers.size(); ++idx)
                layers[idx].save(path + "_l" + std::to_string(idx), filetype);
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
            embedding.load(path + "_e", filetype, stream);
            for (core::index_type idx = 0; idx < layers.size(); ++idx)
                layers[idx].load(path + "_l" + std::to_string(idx), filetype,
                                 stream);
        }
    };

}  // namespace layers
