/**
 * @file color.cuh
 * @brief Implements ANSI constants for coloring the output stream.
 *
 * This file is part of the `test` namespace, which implements classes and
 * functions for testing.
 */

#pragma once

#include <string>

/**
 * @internal
 *
 * @namespace test
 * @brief This namespace contains functions and classes for testing.
 *
 * This specific file defines ANSI constants for coloring the output stream.
 *
 * @endinternal
 */
namespace test {

    /**
     * @namespace color
     * @brief This namespace contains ANSI constants for coloring the output
     * stream.
     *
     * @note This namespace is part of the `test` namespace.
     */
    namespace color {

        const std::string black = "\x1b[30m";
        const std::string red = "\x1b[31m";
        const std::string green = "\x1b[32m";
        const std::string yellow = "\x1b[33m";
        const std::string blue = "\x1b[34m";
        const std::string magenta = "\x1b[35m";
        const std::string cyan = "\x1b[36m";
        const std::string white = "\x1b[37m";
        const std::string gray = "\33[90m";
        const std::string bold = "\33[1m";
        const std::string underline = "\33[4m";
        const std::string reset = "\x1b[39m\33[0m";

    }  // namespace color

}  // namespace test
