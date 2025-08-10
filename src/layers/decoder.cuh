/**
 * @file decoder.cuh
 * @brief Implements the Decoder class.
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

#include "../activations/softmax.cuh"
#include "../core.cuh"
#include "decoderlayer.cuh"
#include "dense.cuh"
#include "embedding.cuh"
#include "layer.cuh"
#include "positionalencoding.cuh"

/**
 * @internal
 *
 * @namespace layers
 * @brief Implements differentiable layers.
 *
 * This specific file defines the Decoder class.
 *
 * @endinternal
 */
namespace layers {

    /**
     * @class Decoder
     * @brief The Decoder layer.
     *
     * The Decoder consists of a stack of DecoderLayers. This is a key component
     * of the Transformer architecture. The output of the Decoder are the
     * probabilities of each token in the vocabulary being the next token in the
     * sequence.
     *
     * For more information, see "Attention Is All You Need".
     *
     * @tparam input_type The data type of the main input tensor elements, i.e.,
     * the type of the target tensor.
     * @tparam output_type The data type of the output tensor elements. (It is
     * also the type of the source input tensor.)
     * @tparam optimizer_type The optimizer used for the parameter updates.
     */
    template <core::integer_type input_type, core::float_type output_type,
              template <typename> class optimizer_type>
    class Decoder final
        : public Layer<input_type, output_type, optimizer_type> {
      private:
        core::Device &device;
        Embedding<input_type, output_type, optimizer_type> embedding;
        PositionalEncoding<output_type> positionalencoding;
        std::vector<DecoderLayer<output_type, optimizer_type>> layers;
        Dense<output_type, optimizer_type> dense;
        activations::Softmax<output_type> softmax;
        core::Device_Tensor<output_type> source_gradient;
        core::Device_Tensor<input_type> null_tensor;

      public:
        /**
         * @brief Construct a new Decoder object.
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param target_shape The shape (excluding the first axis) of the main
         * input, i.e., `[target_sequence_length]`.
         * @param source_shape The shape (excluding the first axis) of the
         * second input, i.e., `[source_sequence_length, embedding_dim]`.
         * @param hidden_shape The shape (excluding the first axis) of the
         * features in the hidden dimension of the Positionwise Feed Forward
         * layer, i.e., `[target_sequence_length, hidden_dim]`.
         * @param num_layers The number of DecoderLayers.
         * @param num_embeddings The size of the dictionary, i.e., this value
         * corresponds to one larger than the largest value in the target
         * tensor.
         * @param optimizer_kw Will be passed to the optimizer. See the
         * documentation of the used optimizer to see which parameters are
         * possible.
         * @param num_heads Number of parallel attention heads.
         * @param dropout The probability that an element in the dropout layer
         * is set to zero.
         * @param max_batchsize The maximum batchsize to be expected.
         * @param stream The CUDA stream used for launching the kernels.
         *
         * @note The output_shape of the layer is automatically derived from the
         * other arguments. It will be
         * `[batchsize, target_sequence_length, num_embeddings]`.
         *
         * @throws std::invalid_argument if `num_layers` is smaller than 1.
         */
        Decoder(core::Device &device,
                const std::vector<core::index_type> &target_shape,
                const std::vector<core::index_type> &source_shape,
                const std::vector<core::index_type> &hidden_shape,
                const core::index_type num_layers,
                const input_type num_embeddings,
                const std::unordered_map<std::string, output_type>
                    &optimizer_kw = {},
                const core::index_type num_heads = 8,
                const output_type dropout = 0.1,
                const core::index_type max_batchsize = 5000,
                const cudaStream_t stream = core::default_stream)
            : device{device},
              embedding{device,
                        target_shape,
                        extend_shape(target_shape, source_shape.back()),
                        num_embeddings,
                        optimizer_kw,
                        sqrt(source_shape.back()),
                        stream},
              positionalencoding{device,
                                 extend_shape(target_shape,
                                              source_shape.back()),
                                 dropout,
                                 max_batchsize,
                                 stream},
              dense{device,
                    extend_shape(target_shape, source_shape.back()),
                    extend_shape(target_shape, num_embeddings),
                    optimizer_kw,
                    true,
                    stream},
              softmax{device,
                      extend_shape(target_shape, num_embeddings),
                      stream},
              source_gradient{core::NO_ALLOC, source_shape} {
            if (num_layers < 1)
                throw std::invalid_argument(
                    "num_layers needs to be at least 1!");

            layers.reserve(num_layers);
            for (core::index_type idx = 0; idx < num_layers; ++idx)
                layers.emplace_back(
                    device, extend_shape(target_shape, source_shape.back()),
                    source_shape, hidden_shape, optimizer_kw, num_heads,
                    dropout, stream);
        }

#pragma diag_suppress virtual_function_decl_hidden

        /**
         * @brief Compute the forward pass of the layer.
         *
         * @note This function will also apply Dropout in training mode.
         *
         * @param target The main input tensor. It has to have the shape
         * `[batchsize, target_sequence_length]`.
         * @param target_mask The target attention mask, i.e., the mask which
         * specifies which positions in the sequence get ignored in the
         * attention calculation. It has to have the shape
         * `[batchsize, target_sequence_length, target_sequence_length]` or
         * `[batchsize, target_sequence_length]`.
         * @param source The output of the Encoder layer. It has to have the
         * shape `[batchsize, source_sequence_length, embedding_dim]`.
         * @param source_mask The source attention mask, i.e., the mask which
         * specifies which positions in the sequence get ignored in the
         * attention calculation. It has to have the shape
         * `[batchsize, source_sequence_length, source_sequence_length]` or
         * `[batchsize, source_sequence_length]`.
         * @param stream The CUDA stream used for launching the kernels.
         * @return The output tensor.
         */
        const core::Device_Tensor<output_type> &forward(
            const core::Device_Tensor<input_type> &target,
            const core::Device_Tensor<core::mask_type> &target_mask,
            const core::Device_Tensor<output_type> &source,
            const core::Device_Tensor<core::mask_type> &source_mask,
            const cudaStream_t stream = core::default_stream) {
            const auto &x1 = embedding.forward(target, stream);
            const auto &x2 = positionalencoding.forward(x1, stream);

            layers.front().forward(x2, target_mask, source, source_mask,
                                   stream);

            for (core::index_type idx = 1; idx < layers.size(); ++idx) {
                const auto &x3 = layers[idx - 1].get_output();
                layers[idx].forward(x3, target_mask, source, source_mask,
                                    stream);
            }

            const auto &x4 = layers.back().get_output();
            const auto &x5 = dense.forward(x4, stream);
            return softmax.forward(x5, stream);
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
         * `input_gradient`. The reason for this is that the forward pass has
         * multiple inputs.
         */
        const core::Device_Tensor<input_type> &backward(
            const core::Device_Tensor<output_type> &error,
            const cudaStream_t stream = core::default_stream) override {
            // calculate gradients until last DecoderLayer
            const auto &x1 = softmax.backward(error, stream);
            const auto &x2 = dense.backward(x1, stream);

            layers.back().backward(x2, stream);

            const auto &y1 = layers.back().get_source_gradient();
            source_gradient.copy_from(y1, stream);

            for (core::index_type idx = 1; idx < layers.size(); ++idx) {
                const core::index_type _idx = layers.size() - idx - 1;
                const auto &x3 = layers[_idx + 1].get_target_gradient();
                layers[_idx].backward(x3, stream);

                const auto &y2 = layers[_idx].get_source_gradient();
                core::add<output_type>
                    <<<device.blocks(source_gradient.size()), device.threads(),
                       0, stream>>>(source_gradient, y2, source_gradient);
                core::checkCuda(cudaPeekAtLastError());
            }

            const auto &x4 = layers.front().get_target_gradient();
            const auto &x5 = positionalencoding.backward(x4, stream);
            embedding.backward(x5, stream);

            return null_tensor;
        }

        /**
         * @brief Get the output of the forward pass.
         */
        const core::Device_Tensor<output_type> &get_output() const override {
            return softmax.get_output();
        }

        /**
         * @brief Get the in the forward pass calculated attention.
         */
        const core::Device_Tensor<output_type> &get_attention() const {
            return layers.back().get_attention();
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
         * @brief Get the gradient with respect to the source.
         */
        const core::Device_Tensor<output_type> &get_source_gradient() const {
            return source_gradient;
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
         * @brief Get all the DecoderLayers as a std::vector.
         */
        const std::vector<DecoderLayer<output_type, optimizer_type>>
            &get_layers() const {
            return layers;
        }

        /**
         * @brief Get a specific DecoderLayer.
         */
        const DecoderLayer<output_type, optimizer_type> &get_layer(
            const core::index_type idx) const {
            return layers[idx];
        }

        /**
         * @brief Get the Dense layer.
         */
        const Dense<output_type, optimizer_type> &get_dense() const {
            return dense;
        }

        /**
         * @brief Get the Softmax function.
         */
        const activations::Softmax<output_type> &get_softmax() const {
            return softmax;
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
            dense.train();
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
            dense.eval();
        }

        /**
         * @brief Returns the number of trainable parameters of the layer.
         */
        core::index_type parameters() const override {
            core::index_type parameters = 0;
            parameters += embedding.parameters();
            for (auto &layer : layers) parameters += layer.parameters();
            parameters += dense.parameters();
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
            size += dense.mem_size();
            size += softmax.mem_size();
            size += source_gradient.mem_size();
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
            embedding.save(path + "_e", filetype);
            for (core::index_type idx = 0; idx < layers.size(); ++idx)
                layers[idx].save(path + "_l" + std::to_string(idx), filetype);
            dense.save(path + "_d", filetype);
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
            dense.load(path + "_d", filetype, stream);
        }
    };

}  // namespace layers
