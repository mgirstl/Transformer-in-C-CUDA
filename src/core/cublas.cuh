/**
 * @file cublas.cuh
 * @brief This file contains wrappers around useful cuBLAS functions for easier
 * usage.
 *
 * The wrappers are needed since tensors are saved in row-major-order in this
 * project but cuBLAS usually uses col-major-order.
 *
 * This file is part of the `core` namespace, which provides fundamental types,
 * utilities, and functions that are widely used across different modules of
 * the project.
 */

#pragma once

#include <cublas_v2.h>

#include "cuda.cuh"
#include "device_tensor.cuh"
#include "types.cuh"

/**
 * @internal
 *
 * @namespace core
 * @brief Provides core facilities used throughout the entire codebase.
 *
 * The `core` namespace contains fundamental types, utilities, and functions
 * that are widely used across different modules of the project.
 *
 * This specific file defines wrappers for some cuBLAS functions.
 *
 * @endinternal
 */
namespace core {

    /**
     * @brief Performs matrix multiplication using the cuBLAS library.
     *
     * The formula used for the matrix multiplication is:
     * @f[ C = \alpha A \cdot B + \beta C @f]
     *
     * This function performs matrix multiplication of two input matrices and
     * stores the result in the output matrix. It uses the appropriate cuBLAS
     * function based on the data type of the matrices.
     *
     * @param handle The handle to the cuBLAS library context.
     * @param alpha The scalar multiplier for the product of the input matrices.
     * @param matrix_a The first input matrix stored in row-major-order.
     * @param col_a The number of columns in the first input matrix.
     * @param trans_a The operation to be performed on the first input matrix
     * (`CUBLAS_OP_N` for normal order, `CUBLAS_OP_T` for transposed order).
     * @param matrix_b The second input matrix stored in row-major-order.
     * @param col_b The number of columns in the second input matrix.
     * @param trans_b The operation to be performed on the second input matrix
     * (`CUBLAS_OP_N` for normal order, `CUBLAS_OP_T` for transposed order).
     * @param beta The scalar multiplier for the output matrix.
     * @param output The output matrix to store the result in row-major-order.
     * @param row_output The number of rows in the output matrix.
     * @param dim_in_between The size of the common dimension between the input
     * matrices.
     * @param col_output The number of columns in the output matrix.
     * @return The status of the cuBLAS operation.
     */
    template <float_type dtype>
    cublasStatus_t inline matrix_multiplication(
        cublasHandle_t &handle, const dtype alpha,
        const Device_Tensor<dtype> &matrix_a, const index_type col_a,
        const cublasOperation_t trans_a, const Device_Tensor<dtype> &matrix_b,
        const index_type col_b, const cublasOperation_t trans_b,
        const dtype beta, Device_Tensor<dtype> &output,
        const index_type row_output, const index_type dim_in_between,
        const index_type col_output) {
        cublasStatus_t error;

        if constexpr (std::is_same_v<dtype, float>)
            error = checkCuda(cublasSgemm(
                handle, trans_b, trans_a, col_output, row_output,
                dim_in_between, &alpha, matrix_b.data(), col_b, matrix_a.data(),
                col_a, &beta, output.data(), col_output));

        else if constexpr (std::is_same_v<dtype, double>)
            error = checkCuda(cublasDgemm(
                handle, trans_b, trans_a, col_output, row_output,
                dim_in_between, &alpha, matrix_b.data(), col_b, matrix_a.data(),
                col_a, &beta, output.data(), col_output));

        else
            static_assert("Unsupported dtype for core::matrix_multiplication!");

        checkCuda(cudaPeekAtLastError());
        return error;
    }

    /**
     * @brief Performs strided batched matrix multiplication using the cuBLAS
     * library.
     *
     * The formula used for the matrix multiplication is:
     * @f[ C' = \alpha A' \cdot B' + \beta C' @f]
     * where @f$ X' = X + j \cdot \mathrm{stride\_X} @f$ for all
     * @f$ j \in [0, 1, ..., \mathrm{batch\_count}-1] @f$
     *
     * This function performs matrix multiplication of two input matrices and
     * stores the result in the output matrix. It uses the appropriate cuBLAS
     * function based on the data type of the matrices.
     *
     * @param handle The handle to the cuBLAS library context.
     * @param alpha The scalar multiplier for the product of the input matrices.
     * @param matrix_a The first input matrix stored in row-major-order.
     * @param col_a The number of columns in the first input matrix.
     * @param stride_a The stride between matrices in the batch for matrix A.
     * @param trans_a The operation to be performed on the first input matrix
     * (`CUBLAS_OP_N` for normal order, `CUBLAS_OP_T` for transposed order).
     * @param matrix_b The second input matrix stored in row-major-order.
     * @param col_b The number of columns in the second input matrix.
     * @param stride_b The stride between matrices in the batch for matrix B.
     * @param trans_b The operation to be performed on the second input matrix
     * (`CUBLAS_OP_N` for normal order, `CUBLAS_OP_T` for transposed order).
     * @param beta The scalar multiplier for the output matrix.
     * @param output The output matrix to store the result in row-major-order.
     * @param row_output The number of rows in the output matrix.
     * @param dim_in_between The size of the common dimension between the input
     * matrices.
     * @param col_output The number of columns in the output matrix.
     * @param stride_c The stride between matrices in the batch for matrix C.
     * @param batch_count The number of matrices in the batch.
     * @return The status of the cuBLAS operation.
     */
    template <float_type dtype>
    cublasStatus_t inline strided_batched_matrix_multiplication(
        cublasHandle_t &handle, const dtype alpha,
        const Device_Tensor<dtype> &matrix_a, const index_type col_a,
        const index_type stride_a, const cublasOperation_t trans_a,
        const Device_Tensor<dtype> &matrix_b, const index_type col_b,
        const index_type stride_b, const cublasOperation_t trans_b,
        const dtype beta, Device_Tensor<dtype> &output,
        const index_type row_output, const index_type dim_in_between,
        const index_type col_output, const index_type stride_c,
        const index_type batch_count) {
        cublasStatus_t error;

        if constexpr (std::is_same_v<dtype, float>)
            error = checkCuda(cublasSgemmStridedBatched(
                handle, trans_b, trans_a, col_output, row_output,
                dim_in_between, &alpha, matrix_b.data(), col_b, stride_b,
                matrix_a.data(), col_a, stride_a, &beta, output.data(),
                col_output, stride_c, batch_count));

        else if constexpr (std::is_same_v<dtype, double>)
            error = checkCuda(cublasDgemmStridedBatched(
                handle, trans_b, trans_a, col_output, row_output,
                dim_in_between, &alpha, matrix_b.data(), col_b, stride_b,
                matrix_a.data(), col_a, stride_a, &beta, output.data(),
                col_output, stride_c, batch_count));

        else
            static_assert(
                "Unsupported dtype for core::strided_batched_matrix_multiplication!");

        checkCuda(cudaPeekAtLastError());
        return error;
    }

    /**
     * @brief Performs matrix-vector multiplication using the cuBLAS library.
     *
     * The formula used for the matrix-vector multiplication is:
     * @f[ \vec{y} = \alpha A \cdot \vec{x} + \beta \cdot \vec{y} @f]
     *
     * This function performs matrix-vector multiplication of an input matrix
     * and an input vector, and stores the result in the output vector. It uses
     * the appropriate cuBLAS function based on the data type of the inputs.
     *
     * @param handle The handle to the cuBLAS library context.
     * @param alpha The scalar multiplier for the product of the input matrix
     * and vector.
     * @param matrix The input matrix stored in row-major-order.
     * @param row The number of rows in the input matrix.
     * @param col The number of columns in the input matrix.
     * @param trans The operation to be performed on the input matrix
     * (`CUBLAS_OP_N` for normal order, `CUBLAS_OP_T` for transposed order).
     * @param vector The input vector.
     * @param beta The scalar multiplier for the output vector.
     * @param output The output vector to store the result.
     * @return The status of the cuBLAS operation.
     */
    template <float_type dtype>
    cublasStatus_t inline matrix_vector_multiplication(
        cublasHandle_t &handle, const dtype alpha,
        const Device_Tensor<dtype> &matrix, const index_type row,
        const index_type col, const cublasOperation_t trans,
        const Device_Tensor<dtype> &vector, const dtype beta,
        Device_Tensor<dtype> &output) {
        cublasStatus_t error;
        cublasOperation_t trans_opposite =
            (trans == CUBLAS_OP_N) ? CUBLAS_OP_T : CUBLAS_OP_N;

        if constexpr (std::is_same_v<dtype, float>)
            error = checkCuda(cublasSgemv(
                handle, trans_opposite, col, row, &alpha, matrix.data(), col,
                vector.data(), 1, &beta, output.data(), 1));

        else if constexpr (std::is_same_v<dtype, double>)
            error = checkCuda(cublasDgemv(
                handle, trans_opposite, col, row, &alpha, matrix.data(), col,
                vector.data(), 1, &beta, output.data(), 1));

        else
            static_assert(
                "Unsupported dtype for core::matrix_vector_multiplication!");

        checkCuda(cudaPeekAtLastError());
        return error;
    }

}  // namespace core
