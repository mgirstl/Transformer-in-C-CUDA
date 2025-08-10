/**
 * @file test.cuh
 * @brief Implements a small testing framework.
 *
 * This file is part of the `test` namespace, which implements classes and
 * functions for testing.
 */

#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "color.cuh"

/**
 * @namespace test
 * @brief This namespace contains functions and classes for testing.
 *
 * @internal
 * This specific file defines a small testing framework.
 * @endinternal
 */
namespace test {

    /**
     * @class Test
     * @brief The Test class allows to create a consistent way to print test
     * results to the terminal and path manipulation.
     */
    class Test {
      private:
        size_t passed;
        size_t total;
        std::string temporary_folder;
        std::string data_folder;

      public:
        /**
         * @brief Construct a new Test object using the command line arguments
         * provided by the main function.
         *
         * @note The first argument saved in argv is expected to be arbitrary.
         * The second should be the path to the data directory for the tests and
         * the third should be a path to a temporary directory which can be used
         * by tests for temporary data if needed.
         *
         * Example usage:
         * @code
         * ./program path/to/data/directory path/to/temporary/directory
         * @endcode
         *
         * @param argc The number of command line arguments.
         * @param argv The command line arguments. The first argument will be
         * ignored because it is assumed to be the program name, similar to the
         * `argv` parameter in the `main` function.
         */
        Test(int argc, char *argv[]) : passed{0}, total{0} {
            if (argc < 3) {
                std::cerr << "Please provide the path to the data directory as "
                          << "the first command-line argument and the path to "
                          << "a temporary directory as the second command-line "
                          << "argument."
                          << std::endl;
                std::exit(-1);
            }
            data_folder = std::string(argv[1]);
            temporary_folder = std::string(argv[2]);
        }

        /**
         * @brief Prints the name of the current series of tests.
         *
         * @param name The name of the series of tests which are currently
         * tested.
         */
        void series(const std::string name) const {
            std::cout << color::blue << color::underline << name << ":"
                      << color::reset << std::endl;
        }

        /**
         * @brief Prints the component, which is currently tested.
         *
         * @param name The name of the component which is currently tested.
         */
        void component(const std::string name) const {
            std::cout << "  " << color::underline << color::yellow << name
                      << ":" << color::reset << std::endl;
        }

        /**
         * @brief Prints a given text in gray.
         *
         * @tparam type Any type for which the `operator<<` is defined for
         * `ostream`.
         * @param text The text to be printed.
         */
        template <typename type>
        void comment(const type text) const {
            std::cout << color::gray << "    XXXXXX: " << text << color::reset
                      << std::endl;
        }

        /**
         * @brief Prints a gray line.
         */
        void comment() const {
            std::cout << color::gray
                      << "    XXXXXX: "
                      << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
                      << "XXXXXXXXXXXXXXX"
                      << color::reset << std::endl;
        }

        /**
         * @brief Checks if the condition is true and prints the corresponding
         * output with a message.
         *
         * @param condition The condition which will be checked.
         * @param msg The message which should be printed, e.g., the name of the
         * current test.
         */
        void check(const bool condition, const std::string msg) {
            ++total;

            if (condition) {
                std::cout << color::green << "    PASSED: " << msg;
                ++passed;
            } else {
                std::cout << color::red << "    FAILED: " << msg;
            }

            std::cout << color::reset << std::endl;
        }

        /**
         * @brief Prints the total number of tests which passed.
         *
         * @return The number of tests which failed.
         */
        int eval() const {
            std::cout << color::blue << "--- " << passed << "/" << total
                      << " checks passed ---" << color::reset << std::endl;

            return total - passed;
        }

        /**
         * @brief Returns the path to a file called name in the data directory
         * specified by `data_folder`.
         *
         * @param name The name of the file.
         * @return The path to the file.
         */
        std::string data_path(const std::string name) const {
            return data_folder + "/" + name;
        }

        /**
         * @brief Returns the path to a file called name in the in the temporary
         * directory specified by `temporary_folder`.
         *
         * @param name The name of the file.
         * @return The path to the file.
         */
        std::string temporary_path(const std::string name) const {
            return temporary_folder + "/" + name;
        }
    };

}  // namespace test
