/**
 * @file upscale.cuh
 * @brief Implements the Upscale class.
 *
 * This file is part of the `layers` namespace, which implements differentiable
 * layers.
 */

#pragma once

#include <stdexcept>
#include <vector>

#include "../core.cuh"
#include "../optimizers/none.cuh"
#include "layer.cuh"

/**
 * @internal
 *
 * @namespace layers
 * @brief Implements differentiable layers.
 *
 * This specific file defines the Upscale class.
 *
 * @endinternal
 */
namespace layers {

    /**
     * @brief This kernel implements bilinear interpolation.
     *
     * The interpolation will be performed for each sample, i.e., for the last
     * two axes of the input tensor.
     *
     * @param input The input tensor to be interpolated. It must have the shape
     * @f$ [K, N, M] @f$.
     * @param output The output tensor to store the results. It must have the
     * shape @f$ [K, N', M'] @f$, where @f$ N' \geq N @f$ and
     * @f$ M' \geq M @f$.
     */
    template <core::arithmetic_type data_type>
    __global__ void upscale_forward(const core::Kernel_Tensor<data_type> input,
                                    core::Kernel_Tensor<data_type> output) {
        const core::index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
        const core::index_type idx_stride = gridDim.x * blockDim.x;
        const core::index_type row = blockIdx.y * blockDim.y + threadIdx.y;
        const core::index_type row_stride = gridDim.y * blockDim.y;
        const core::index_type col = blockIdx.z * blockDim.z + threadIdx.z;
        const core::index_type col_stride = gridDim.z * blockDim.z;

        const core::index_type batchsize = input.shape(0);

        const core::index_type input_image_height = input.shape(1);
        const core::index_type input_image_width = input.shape(2);
        const core::index_type input_image_size =
            input_image_height * input_image_width;

        const core::index_type output_image_height = output.shape(1);
        const core::index_type output_image_width = output.shape(2);
        const core::index_type output_image_size =
            output_image_height * output_image_width;

        const data_type r_scale =
            data_type(output_image_height - 1) / (input_image_height - 1);
        const data_type c_scale =
            data_type(output_image_width - 1) / (input_image_width - 1);

        for (core::index_type i = idx; i < batchsize; i += idx_stride) {
            for (core::index_type r = row; r < output_image_height;
                 r += row_stride) {
                const core::index_type r_lower = r / r_scale;
                const core::index_type r_higher =
                    min(r_lower + 1, input_image_height - 1);

                for (core::index_type c = col; c < output_image_width;
                     c += col_stride) {
                    const core::index_type c_lower = c / c_scale;
                    const core::index_type c_higher =
                        min(c_lower + 1, input_image_width - 1);

                    const data_type input_00 =
                        input[i * input_image_size +
                              r_lower * input_image_width + c_lower];
                    const data_type input_01 =
                        input[i * input_image_size +
                              r_lower * input_image_width + c_higher];
                    const data_type input_10 =
                        input[i * input_image_size +
                              r_higher * input_image_width + c_lower];
                    const data_type input_11 =
                        input[i * input_image_size +
                              r_higher * input_image_width + c_higher];

                    const data_type r_ratio = r / r_scale - r_lower;
                    const data_type c_ratio = c / c_scale - c_lower;

                    output[i * output_image_size + r * output_image_width + c] =
                        (1 - r_ratio) *
                            ((1 - c_ratio) * input_00 + c_ratio * input_01) +
                        r_ratio *
                            ((1 - c_ratio) * input_10 + c_ratio * input_11);
                }
            }
        }
    }

    /**
     * @class Upscale
     * @brief Upscale Layer using bilinear interpolation.
     *
     * This layer scales a tensor of shape @f$ [K, N, M] @f$ up to a shape
     * @f$ [K, N', M'] @f$ using bilinear interpolation, where @f$ N' \geq N @f$
     * and @f$ M' \geq M @f$.
     *
     * @tparam data_type The data type of the tensor elements.
     */
    template <core::float_type data_type>
    class Upscale final : public Layer<data_type, data_type, optimizers::None> {
      private:
        const core::Device &device;
        std::vector<core::index_type> output_shape;
        core::Device_Tensor<data_type> output;

      public:
        /**
         * @brief Construct a new Upscale object.
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param output_shape The shape (excluding the first axis) of the
         * output tensors.
         */
        Upscale(const core::Device &device,
                const std::vector<core::index_type> &output_shape)
            : device{device},
              output_shape{output_shape},
              output{core::NO_ALLOC, output_shape} {}

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
            output.rebatchsize(input.batchsize(), stream);

            upscale_forward<data_type>
                <<<device.blocks_3D(input.batchsize(), output_shape[0],
                                    output_shape[1]),
                   device.threads_3D(), 0, stream>>>(input, output);
            core::checkCuda(cudaPeekAtLastError());

            return output;
        }

        /**
         * @brief Compute the backward pass of the layer.
         *
         * @throws std::runtime_error
         */
        const core::Device_Tensor<data_type> &backward(
            const core::Device_Tensor<data_type> & /* unused */,
            const cudaStream_t /* unused */ = core::default_stream) override {
            throw std::runtime_error(
                "The backward pass of this layer is not implemented!");
        }

        /**
         * @brief Get the output of the forward pass.
         */
        const core::Device_Tensor<data_type> &get_output() const override {
            return output;
        }

        /**
         * @brief Get the gradient in respect to the input.
         *
         * @throws std::runtime_error
         */
        const core::Device_Tensor<data_type> &get_input_gradient()
            const override {
            throw std::runtime_error(
                "The backward pass of this layer is not implemented and hence "
                "the input_gradient not computed!");
        }

        /**
         * @brief Returns the memory footprint of the object on GPU.
         */
        const core::Mem_Size mem_size(
            const bool /* unused */ = false) const override {
            return output.mem_size();
        }
    };

}  // namespace layers
