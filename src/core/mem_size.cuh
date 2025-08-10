/**
 * @file mem_size.cuh
 * @brief This file defines the struct Mem_Size.
 *
 * This file is part of the `core` namespace, which provides fundamental types,
 * utilities, and functions that are widely used across different modules of
 * the project.
 */

#pragma once

/**
 * @internal
 *
 * @namespace core
 * @brief Provides core facilities used throughout the entire codebase.
 *
 * The `core` namespace contains fundamental types, utilities, and functions
 * that are widely used across different modules of the project.
 *
 * This specific file defines the Mem_Size struct.
 *
 * @endinternal
 */
namespace core {

    /**
     * @struct Mem_Size
     * @brief The Mem_Size class is used to save the memory footprint of objects
     * on GPU.
     *
     * The `fixed_size` is the size of an object which is independent of the
     * batch size of the neural network input. The `variable_size` is the size
     * which still needs to be multiplied by the batch size of the network's
     * input. The reason for this distinction is that the shape of some tensors,
     * and thus their size, will change depending on the batch size of the
     * network's inputs.
     */
    struct Mem_Size {
        size_t fixed_size;
        size_t variable_size;

        // Constructor
        Mem_Size(size_t fixed_size = 0, size_t variable_size = 0)
            : fixed_size{fixed_size}, variable_size{variable_size} {}

        // Define operator+
        Mem_Size operator+(const Mem_Size &other) const {
            return {fixed_size + other.fixed_size,
                    variable_size + other.variable_size};
        }

        // Define operator+=
        Mem_Size &operator+=(const Mem_Size &other) {
            fixed_size += other.fixed_size;
            variable_size += other.variable_size;
            return *this;
        }
    };

}  // namespace core
