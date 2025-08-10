/**
 * @file transformer.cuh
 * @brief Implements the class Transformer.
 *
 * This file is part of the `models` namespace, which implements neural network
 * models.
 */

#pragma once

#include <sstream>
#include <string>
#include <vector>

#include "../core.cuh"
#include "../layers.cuh"
#include "../utils.cuh"
#include "classifier.cuh"
#include "info.cuh"

/**
 * @internal
 *
 * @namespace models
 * @brief Implements neural network models.
 *
 * This specific file defines the Transformer class.
 *
 * @endinternal
 */
namespace models {

    /**
     * @class Transformer
     * @brief A transformer network which can be trained on a single input
     * vector.
     *
     * @tparam input_type The type of the input and labels of the network.
     * @tparam output_type The type used for calculations inside of the network
     * and the output.
     * @tparam optimizer_type The optimizer used for the parameter updates.
     */
    template <core::integer_type input_type, core::float_type output_type,
              template <typename> class optimizer_type>
    class Transformer final : public Classifier<input_type, output_type,
                                                input_type, optimizer_type> {
      protected:
        const utils::Config<output_type> &config;
        layers::Transformer<input_type, output_type, optimizer_type>
            transformer;

      public:
        /**
         * @brief Construct a new Transformer object.
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param config The config used for creating the network.
         */
        Transformer(core::Device &device,
                    const utils::Config<output_type> &config)
            : config{config},
              Classifier<input_type, output_type, input_type, optimizer_type>{
                  device, config, config.num_embeddings, config.sequence_length,
                  config.ignore_index},
              transformer{device,
                          {config.sequence_length},
                          {config.sequence_length},
                          config.embedding_dim,
                          config.hidden_dim,
                          config.num_encoder_layers,
                          config.num_decoder_layers,
                          config.num_embeddings,
                          config.num_embeddings,
                          {{"learning_rate", config.learning_rate},
                           {"model_dim", config.embedding_dim},
                           {"warmup_steps", config.noam_warmup_steps}},
                          config.num_heads,
                          config.dropout,
                          config.ignore_index,
                          config.max_batchsize} {}

        /**
         * @brief Returns the number of trainable parameters of the classifier.
         */
        core::index_type parameters() const override {
            core::index_type parameters = 0;
            parameters += transformer.parameters();
            return parameters;
        }

        /**
         * @brief Returns the memory footprint of the object on GPU.
         */
        const core::Mem_Size mem_size(
            const bool /* unused */ = false) const override {
            core::Mem_Size size;
            size += Classifier<input_type, output_type, input_type,
                               optimizer_type>::mem_size();
            size += transformer.mem_size();
            return size;
        }

        /**
         * @brief Returns information about the network.
         */
        const std::string info() const override {
            std::ostringstream oss;

            info::format(oss, "Transformer");
            info::line(oss);
            info::format(oss, "Classifier",
                         Classifier<input_type, output_type, input_type,
                                    optimizer_type>::mem_size());
            info::format(oss, "Transformer", transformer.mem_size(),
                         transformer.parameters());

            // Encoder
            info::format(oss, "  + Encoder",
                         transformer.get_encoder().mem_size(),
                         transformer.get_encoder().parameters());
            info::format(
                oss, "    - Embedding",
                transformer.get_encoder().get_embedding().mem_size(),
                transformer.get_encoder().get_embedding().parameters());
            info::format(
                oss, "    - Positional Encoding",
                transformer.get_encoder().get_positionalencoding().mem_size(),
                transformer.get_encoder()
                    .get_positionalencoding()
                    .parameters());
            info::format(oss,
                         "    - Encoder Layer (" +
                             std::to_string(config.num_encoder_layers) + "x)",
                         transformer.get_encoder().get_layer(0).mem_size(),
                         transformer.get_encoder().get_layer(0).parameters());

            // Decoder
            info::format(oss, "  + Decoder",
                         transformer.get_decoder().mem_size(),
                         transformer.get_decoder().parameters());
            info::format(
                oss, "    - Embedding",
                transformer.get_decoder().get_embedding().mem_size(),
                transformer.get_decoder().get_embedding().parameters());
            info::format(
                oss, "    - Positional Encoding",
                transformer.get_decoder().get_positionalencoding().mem_size(),
                transformer.get_decoder()
                    .get_positionalencoding()
                    .parameters());
            info::format(oss,
                         "    - Decoder Layer (" +
                             std::to_string(config.num_decoder_layers) + "x)",
                         transformer.get_decoder().get_layer(0).mem_size(),
                         transformer.get_decoder().get_layer(0).parameters());
            info::format(oss, "    - Dense",
                         transformer.get_decoder().get_dense().mem_size(),
                         transformer.get_decoder().get_dense().parameters());
            info::format(oss, "    - Softmax",
                         transformer.get_decoder().get_softmax().mem_size());

            // Total
            info::double_line(oss);
            info::format(oss, "Total:", mem_size(), parameters());
            info::format(oss, "Batchsize: " + std::to_string(config.batchsize),
                         {mem_size().fixed_size,
                          config.batchsize * mem_size().variable_size},
                         parameters(), false);

            return oss.str();
        }

      private:
        /**
         * @brief Compute the forward pass of the network.
         *
         * @param input The input tensor.
         * @param stream The CUDA stream to use for computation.
         * @return The output tensor.
         */
        const core::Device_Tensor<output_type> &forward(
            const core::Device_Tensor<input_type> &input,
            const cudaStream_t stream) override {
            return transformer.forward(input, input, stream);
        };

        /**
         * @brief Compute the backward pass of the network.
         *
         * @param error The error tensor from the loss function.
         * @param stream The CUDA stream to use for computation.
         */
        void backward(const core::Device_Tensor<output_type> &error,
                      const cudaStream_t stream) {
            transformer.backward(error, stream);
        }

        /**
         * @brief Sets the network into training mode.
         *
         * This changes the calculation in the forward pass for some layers,
         * e.g., Dropout. Moreover, in training mode, the parameter update
         * step will be performed when calling the backward pass of the layer.
         *
         * In general, layers are initialized in training mode.
         */
        void _train() override { transformer.train(); }

        /**
         * @brief Sets the network into evaluation mode.
         *
         * This changes the calculation in the forward pass for some layers,
         * e.g., Dropout. Moreover, in evaluation mode, no parameter update
         * is performed in the backward pass.
         *
         * In general, layers are initialized in training mode.
         */
        void _eval() override { transformer.eval(); }

        /**
         * @brief Saves the parameters of the classifier to a file.
         *
         * @param path The base path to write the parameters to.
         * @param filetype The filetype of the saved files.
         */
        void _save(const std::string &path,
                   const std::string &filetype) override {
            transformer.save(path, filetype);
        }

        /**
         * @brief Loads the parameters of the classifier from a file.
         *
         * @param path The base path to read the parameters from.
         * @param filetype The filetype of the saved files.
         * @param stream The CUDA stream used for copying from the host.
         */
        void _load(const std::string &path, const std::string &filetype,
                           const cudaStream_t &stream) override {
            transformer.load(path, filetype, stream);
        }
    };

}  // namespace models
