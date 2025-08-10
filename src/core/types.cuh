/**
 * @file types.cuh
 * @brief This file contains often used type definitions.
 *
 * This file is part of the `core` namespace, which provides fundamental types,
 * utilities, and functions that are widely used across different modules of
 * the project.
 */

#pragma once

#include <type_traits>

/**
 * @internal
 *
 * @namespace core
 * @brief Provides core facilities used throughout the entire codebase.
 *
 * The `core` namespace contains fundamental types, utilities, and functions
 * that are widely used across different modules of the project.
 *
 * This specific file defines some commonly used types.
 *
 * @endinternal
 */
namespace core {

    // Basic data types
    using index_type = unsigned long long;
    using data_type = float;
    using label_type = int;
    using mask_type = bool;

    /**
     * @concept arithmetic_type
     * @brief Concept that checks if a type is arithmetic.
     */
    template <typename T>
    concept arithmetic_type = std::is_arithmetic_v<T>;

    /**
     * @concept float_type
     * @brief Concept that checks if a type is a floating-point type.
     */
    template <typename T>
    concept float_type = std::is_floating_point_v<T>;

    /**
     * @concept integer_type
     * @brief Concept that checks if a type is an integer type.
     */
    template <typename T>
    concept integer_type = std::is_integral_v<T>;

    /**
     * @struct NO_ALLOC_type
     * @brief A type to indicate that a tensor should not be allocated, i.e., it
     * is a tag type.
     */
    struct NO_ALLOC_type {
        explicit NO_ALLOC_type() = default;
    };
    constexpr NO_ALLOC_type NO_ALLOC;

    /**
     * @struct NO_BATCHSIZE_type
     * @brief A type to indicate that a function should ignore the batch size,
     * it is a tag type.
     */
    struct NO_BATCHSIZE_type {
        explicit NO_BATCHSIZE_type() = default;
    };
    constexpr NO_BATCHSIZE_type NO_BATCHSIZE;

    /**
     * @struct vector_type
     * @brief A template struct that defines vector types based on the provided
     * data type.
     *
     * This struct template is specialized for different data types to define
     * the corresponding vector type and the number of elements per vector. It
     * provides a way to handle different vector types in a generic manner.
     */
    template <typename T>
    struct vector_type;

    /**
     * @struct vector_type<float>
     * @brief Specialization of `vector_type` for the `float` data type.
     *
     * This specialization defines the vector type for `float` as `float4` and
     * specifies that there are 4 elements per vector.
     */
    template <>
    struct vector_type<float> {
        using type = float4;
        static constexpr index_type elements_per_vector = 4;
    };

    /**
     * @struct vector_type<double>
     * @brief Specialization of `vector_type` for the `double` data type.
     *
     * This specialization defines the vector type for `double` as `double2` and
     * specifies that there are 2 elements per vector.
     */
    template <>
    struct vector_type<double> {
        using type = double2;
        static constexpr index_type elements_per_vector = 2;
    };

}  // namespace core
