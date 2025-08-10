/**
 * @file layer.cuh
 * @brief Implements the interface for differentiable layers.
 *
 * This file is part of the `layers` namespace, which implements differentiable
 * layers.
 */

#pragma once

#include <stdexcept>
#include <type_traits>

#include "../core.cuh"
#include "../optimizers/optimizer.cuh"

/**
 * @namespace layers
 * @brief Implements differentiable layers.
 *
 * @internal
 * This specific file defines the Layer interface.
 * @endinternal
 */
namespace layers {

    /**
     * @interface Layer
     * @brief Interface for differentiable layers.
     *
     * This class template provides an interface for differentiable layers in a
     * neural network.
     *
     * @tparam input_type The data type of input tensor elements.
     * @tparam output_type The data type of the output tensor elements.
     * @tparam optimizer_type The optimizer used for the parameter updates.
     *
     * The template parameters need to follow the following rules:
     * - If both `input_type` and `output_type` are integers, they must be the
     * same type.
     * - If both `input_type` and `output_type` are floating-point types, they
     * must be the same type.
     * - The `optimizer_type` needs to fulfill the `optimizers::optimizer_type`
     * concept.
     */
    template <core::arithmetic_type input_type,
              core::arithmetic_type output_type,
              template <typename> class optimizer_type>
    class Layer {
        static_assert((std::is_integral<input_type>::value &&
                       std::is_integral<output_type>::value &&
                       std::is_same<input_type, output_type>::value) ||
                          (std::is_floating_point<input_type>::value &&
                           std::is_floating_point<output_type>::value &&
                           std::is_same<input_type, output_type>::value) ||
                          (std::is_integral<input_type>::value &&
                           std::is_floating_point<output_type>::value) ||
                          (std::is_floating_point<input_type>::value &&
                           std::is_integral<output_type>::value),
                      "input_type and output_type must be the same type if "
                      "they are both integers or both floating-point types.");

        /*
         * Using float for the concept check to ensure optimizer_type satisfies
         * the optimizers::optimizer_type concept. The data type of the tensor
         * elements of the optimizer will be decided by the derived class.
         */
        static_assert(optimizers::optimizer_type<optimizer_type<float>, float>,
                      "optimizer_type must satisfy the "
                      "optimizers::optimizer_type concept!");

      public:
        /**
         * @brief Compute the forward pass of the layer.
         *
         * This function should be overridden by derived classes to implement
         * the specific forward pass logic. If not overridden, it raises a
         * runtime error indicating that additional arguments are required.
         *
         * @param input The input tensor.
         * @param stream The CUDA stream to use for computation.
         * @return The output tensor.
         *
         * @throws std::invalid_argument if the function is not overridden in a
         * derived class.
         */
        virtual const core::Device_Tensor<output_type> &forward(
            const core::Device_Tensor<input_type> &input,
            const cudaStream_t stream) {
            throw std::invalid_argument(
                "This layer requires additional arguments for the forward "
                "pass.");
        }

        /**
         * @brief Compute the backward pass of the layer.
         *
         * This function should be overridden by derived classes to implement
         * the specific backward pass logic. If not overridden, it raises a
         * runtime error indicating that additional arguments are required.
         *
         * @note This function will also update the parameters if the layer is
         * in training mode.
         *
         * @param error The error tensor from the next layer.
         * @param stream The CUDA stream to use for computation.
         * @return The gradient with respect to the input.
         *
         * @throws std::invalid_argument if the function is not overridden in a
         * derived class.
         */
        virtual const core::Device_Tensor<input_type> &backward(
            const core::Device_Tensor<output_type> &error,
            const cudaStream_t stream) {
            throw std::invalid_argument(
                "This layer requires additional arguments for the backward "
                "pass.");
        }

        /**
         * @brief Get the output of the forward pass.
         *
         * This function should be overridden by derived classes if the forward
         * pass calculates a single output.
         *
         * @throws std::runtime_error if the function is not overridden in a
         * derived class.
         */
        virtual const core::Device_Tensor<output_type> &get_output() const {
            throw std::runtime_error(
                "This layer returns multiple outputs in the forward pass!");
        }

        /**
         * @brief Get the gradient with respect to the input.
         *
         * This function should be overridden by derived classes if the backward
         * pass calculates multiple input gradients. A layer calculates multiple
         * input gradients if the forward pass gets multiple differentiable
         * inputs.
         *
         * @throws std::runtime_error if the function is not overridden in a
         * derived class.
         */
        virtual const core::Device_Tensor<input_type> &get_input_gradient()
            const {
            throw std::runtime_error(
                "This layer calculates multiple input gradients in the "
                "backward pass!");
        }

        /**
         * @brief Saves the parameters of the layer to a file.
         *
         * @param path The base path to write the parameters to.
         *
         * @note It is assumed that the filetype is "tensor".
         */
        void save(const std::string &path) { _save(path, "tensor"); }

        /**
         * @brief Saves the parameters of the layer to a file.
         *
         * @param path The base path to write the parameters to.
         * @param filetype The filetype of the saved files.
         */
        void save(const std::string &path, const std::string &filetype) {
            _save(path, filetype);
        }

        /**
         * @brief Loads the parameters of the layer from a file.
         *
         * @param path The base path to read the parameters from.
         *
         * @note It is assumed that the filetype is "tensor". The kernel used
         * to copy from host to device is the `core::default_stream`.
         */
        void load(const std::string &path) {
            _load(path, "tensor", core::default_stream);
        }

        /**
         * @brief Loads the parameters of the layer from a file.
         *
         * @param path The base path to read the parameters from.
         * @param filetype The filetype of the saved files.
         *
         * @note The kernel used to copy from host to device is the
         * `core::default_stream`.
         */
        void load(const std::string &path, const std::string &filetype) {
            _load(path, filetype, core::default_stream);
        }

        /**
         * @brief Loads the parameters of the layer from a file.
         *
         * @param path The base path to read the parameters from.
         * @param stream The CUDA stream used for copying from the host.
         *
         * @note It is assumed that the filetype is "tensor".
         */
        void load(const std::string &path, const cudaStream_t &stream) {
            _load(path, "tensor", stream);
        }

        /**
         * @brief Loads the parameters of the layer from a file.
         *
         * @param path The base path to read the parameters from.
         * @param filetype The filetype of the saved files.
         * @param stream The CUDA stream used for copying from the host.
         */
        void load(const std::string &path, const std::string &filetype,
                  const cudaStream_t &stream) {
            _load(path, filetype, stream);
        }

        /**
         * @brief Sets the layer into training mode.
         *
         * This changes the calculation in the forward pass for some layers,
         * e.g., Dropout. Moreover, in training mode, the parameter update
         * step will be performed when calling the backward pass of the layer.
         *
         * In general, layers are initialized in training mode.
         *
         * @note This function should be overridden by derived classes if
         * the derived class behaves differently in training mode than in
         * evaluation mode.
         */
        virtual void train() {
            // Default implementation does nothing
        }

        /**
         * @brief Sets the layer into evaluation mode.
         *
         * This changes the calculation in the forward pass for some layers,
         * e.g., Dropout. Moreover, in evaluation mode, no parameter update
         * is performed in the backward pass.
         *
         * In general, layers are initialized in training mode.
         *
         * @note This function should be overridden by derived classes if
         * the derived class behaves differently in training mode than in
         * evaluation mode.
         */
        virtual void eval() {
            // Default implementation does nothing
        }

        /**
         * @brief Returns the number of trainable parameters of the layer.
         *
         * @note This function should be overridden by derived classes if
         * the derived class has trainable parameters.
         */
        virtual core::index_type parameters() const { return 0; }

        /**
         * @brief Returns the memory footprint of the object on GPU.
         *
         * @note This function should be overridden by derived classes if
         * the derived class allocates memory on the GPU.
         */
        virtual const core::Mem_Size mem_size(
            const bool /* unused */ = false) const {
            return {0, 0};
        }

        /**
         * @brief Destroy the Layer object.
         *
         * Ensures correct behavior when deleting derived classes.
         */
        virtual ~Layer() {}

      protected:
        /**
         * @brief Saves the parameters of the layer to a file.
         *
         * @param path The base path to write the parameters to.
         * @param filetype The filetype of the saved files.
         *
         * @note This function should be overridden by derived classes if
         * the derived class has trainable parameters.
         */
        virtual void _save(const std::string &path,
                           const std::string &filetype) {
            // Default implementation does nothing
        }

        /**
         * @brief Loads the parameters of the layer from a file.
         *
         * @param path The base path to read the parameters from.
         * @param filetype The filetype of the saved files.
         * @param stream The CUDA stream used for copying from the host.
         *
         * @note This function should be overridden by derived classes if
         * the derived class has trainable parameters.
         */
        virtual void _load(const std::string &path, const std::string &filetype,
                           const cudaStream_t &stream) {
            // Default implementation does nothing
        }
    };

}  // namespace layers
