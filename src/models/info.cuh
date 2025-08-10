/**
 * @file info.cuh
 * @brief Implements some helpful functions to format the info string of the
 * classifiers.
 *
 * This file is part of the `models` namespace, which implements neural network
 * models.
 */

#pragma once

#include <iomanip>
#include <sstream>
#include <string>

#include "../core.cuh"

/**
 * @internal
 *
 * @namespace models
 * @brief Implements neural network models.
 *
 * This specific file defines helpful functions to format the info string of the
 * classifiers.
 *
 * @endinternal
 */
namespace models {

    /**
     * @namespace info
     * @brief Implements some helpful functions to format the info string of the
     * classifiers.
     *
     * @note This namespace is part of the `models` namespace.
     */
    namespace info {

        constexpr core::index_type total_width = 80;
        constexpr core::index_type fixed_size_width = 19;
        constexpr core::index_type variable_size_width = 19;
        constexpr core::index_type trainable_params_width = 16;
        constexpr core::index_type name_width = total_width
                                              - fixed_size_width
                                              - variable_size_width
                                              - trainable_params_width;

        /**
         * @brief Formats the network title.
         *
         * @param oss The string to add the information to.
         * @param title The name of the network.
         * @param new_line If true, then a new line character is added at the
         * end of the string.
         */
        void format(std::ostringstream &oss, const std::string title,
                    const bool new_line = true) {
            oss << std::left << std::setw(name_width) << title << std::right
                << std::setw(fixed_size_width) << "Fixed Size" << std::right
                << std::setw(variable_size_width) << "Variable Size"
                << std::right << std::setw(trainable_params_width)
                << "Parameters";
            if (new_line) oss << "\n";
        }

        /**
         * @brief Formats the information of a network component.
         *
         * @param oss The string to add the information to.
         * @param name The name of the component.
         * @param mem_size The memory size of the component.
         * @param parameters The number of trainable parameters of the
         * component.
         * @param new_line If true, then a new line character is added at the
         * end of the string.
         */
        void format(std::ostringstream &oss, const std::string name,
                    const core::Mem_Size mem_size,
                    const core::index_type parameters = 0,
                    const bool new_line = true) {
            oss << std::fixed << std::setprecision(2) << std::left
                << std::setw(name_width) << name << std::right
                << std::setw(fixed_size_width - 3)
                << (mem_size.fixed_size / (1024. * 1024.)) << " MB"
                << std::right << std::setw(variable_size_width - 3)
                << (mem_size.variable_size / (1024. * 1024.)) << " MB";
            if (parameters)
                oss << std::right << std::setw(trainable_params_width)
                    << parameters;
            if (new_line) oss << "\n";
        }

        /**
         * @brief Adds a single line to oss.
         *
         * @param new_line If true, then a new line character is added at the
         * end of the string.
         */
        void line(std::ostringstream &oss, const bool new_line = true) {
            for (core::index_type idx = 0; idx < total_width; ++idx) oss << "-";
            if (new_line) oss << "\n";
        }

        /**
         * @brief Adds a double line to oss.
         *
         * @param new_line If true, then a new line character is added at the
         * end of the string.
         */
        void double_line(std::ostringstream &oss, const bool new_line = true) {
            for (core::index_type idx = 0; idx < total_width; ++idx) oss << "=";
            if (new_line) oss << "\n";
        }

    }  // namespace info

}  // namespace models
