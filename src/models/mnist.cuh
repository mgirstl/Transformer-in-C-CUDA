/**
 * @file mnist.cuh
 * @brief Implements the class MNIST.
 *
 * This file is part of the `models` namespace, which implements neural network
 * models.
 */

#pragma once

#include <sstream>
#include <string>
#include <vector>

#include "../activations.cuh"
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
 * This specific file defines the MNIST class.
 *
 * @endinternal
 */
namespace models {

    /**
     * @class MNIST
     * @brief A class which can be trained on the MNIST dataset.
     *
     * @tparam data_type The type used for calculations inside of the network
     * and the input.
     * @tparam label_type The type of the labels of the network.
     * @tparam optimizer_type The optimizer used for the parameter updates.
     */
    template <core::float_type data_type, core::integer_type label_type,
              template <typename> class optimizer_type>
    class MNIST final
        : public Classifier<data_type, data_type, label_type, optimizer_type> {
      private:
        const utils::Config<data_type> &config;
        layers::Upscale<data_type> upscale;
        layers::Reshape<data_type> reshape;
        layers::Dense<data_type, optimizer_type> dense_1;
        activations::ReLU<data_type> relu;
        layers::Dense<data_type, optimizer_type> dense_2;
        activations::Softmax<data_type> softmax;

      public:
        /**
         * @brief Construct a new MNIST object.
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param config The config used for creating the network.
         */
        MNIST(core::Device &device, const utils::Config<data_type> &config)
            : config{config},
              Classifier<data_type, data_type, label_type, optimizer_type>{
                  device, config, config.num_classes},
              upscale{device, {config.input_1D, config.input_1D}},
              reshape{{config.input_1D, config.input_1D},
                      {config.input_1D * config.input_1D}},
              dense_1{device,
                      {config.input_1D * config.input_1D},
                      {config.hidden_dim},
                      {{"learning_rate", config.learning_rate}}},
              relu{device, {config.hidden_dim}},
              dense_2{device,
                      {config.hidden_dim},
                      {config.num_classes},
                      {{"learning_rate", config.learning_rate}}},
              softmax{device, {config.num_classes}} {}

        /**
         * @brief Returns the number of trainable parameters of the classifier.
         */
        core::index_type parameters() const override {
            core::index_type parameters = 0;
            parameters += dense_1.parameters();
            parameters += dense_2.parameters();
            return parameters;
        }

        /**
         * @brief Returns the memory footprint of the object on GPU.
         */
        const core::Mem_Size mem_size(
            const bool /* unused */ = false) const override {
            core::Mem_Size size;
            size += Classifier<data_type, data_type, label_type,
                               optimizer_type>::mem_size();
            size += upscale.mem_size();
            size += reshape.mem_size();
            size += dense_1.mem_size();
            size += relu.mem_size();
            size += dense_2.mem_size();
            size += softmax.mem_size();
            return size;
        }

        /**
         * @brief Returns information about the network.
         */
        const std::string info() const override {
            std::ostringstream oss;

            info::format(oss, "MNIST");
            info::line(oss);
            info::format(oss, "Classifier",
                         Classifier<data_type, data_type, label_type,
                                    optimizer_type>::mem_size());
            info::format(oss, "Upscale", upscale.mem_size());
            info::format(oss, "Reshape", reshape.mem_size());
            info::format(oss, "Dense (1)", dense_1.mem_size(),
                         dense_1.parameters());
            info::format(oss, "ReLU", relu.mem_size());
            info::format(oss, "Dense (2)", dense_2.mem_size(),
                         dense_2.parameters());
            info::format(oss, "Softmax", softmax.mem_size());
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
        const core::Device_Tensor<data_type> &forward(
            const core::Device_Tensor<data_type> &input,
            const cudaStream_t stream) override {
            ;
            const auto &x1 = upscale.forward(input, stream);
            const auto &x2 = reshape.forward(x1, stream);
            const auto &x3 = dense_1.forward(x2, stream);
            const auto &x4 = relu.forward(x3, stream);
            const auto &x5 = dense_2.forward(x4, stream);
            return softmax.forward(x5, stream);
        };

        /**
         * @brief Compute the backward pass of the network.
         *
         * @param error The error tensor from the loss function.
         * @param stream The CUDA stream to use for computation.
         */
        void backward(const core::Device_Tensor<data_type> &error,
                      const cudaStream_t stream) {
            const auto &x1 = softmax.backward(error, stream);
            const auto &x2 = dense_2.backward(x1, stream);
            const auto &x3 = relu.backward(x2, stream);
            dense_1.backward(x3, stream);
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
        void _train() override {
            dense_1.train();
            dense_2.train();
        }

        /**
         * @brief Sets the network into evaluation mode.
         *
         * This changes the calculation in the forward pass for some layers,
         * e.g., Dropout. Moreover, in evaluation mode, no parameter update
         * is performed in the backward pass.
         *
         * In general, layers are initialized in training mode.
         */
        void _eval() override {
            dense_1.eval();
            dense_2.eval();
        }

        /**
         * @brief Saves the parameters of the classifier to a file.
         *
         * @param path The base path to write the parameters to.
         * @param filetype The filetype of the saved files.
         */
        void _save(const std::string &path,
                   const std::string &filetype) override {
            dense_1.save(path + "_d1", filetype);
            dense_2.save(path + "_d2", filetype);
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
            dense_1.load(path + "_d1", filetype, stream);
            dense_2.load(path + "_d2", filetype, stream);
        }
    };

}  // namespace models
