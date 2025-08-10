/**
 * @file config.cuh
 * @brief Implements the Config class.
 *
 * This file is part of the `utils` namespace, which implements utility
 * functions and classes.
 */

#pragma once

#include <fstream>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../core.cuh"

/**
 * @namespace utils
 * @brief Namespace for utility functions and classes.
 *
 * @internal
 * This specific file defines the Config class.
 * @endinternal
 */
namespace utils {

    /**
     * @class Config
     * @brief A class to handle configuration settings.
     *
     * Example usage:
     * @code
     * ./program config.txt epochs=2 config2.txt
     * @endcode
     * Here, first `config.txt` is parsed, then `config2.txt`, and afterwards
     * `epochs=2`.
     *
     * @note In a config file, each line starting with a '#' or not containing a
     * '=' gets ignored. The parsing happens in the following way: the '=' will
     * be replaced with a ' '. Then, ignoring leading ' ', the first substring
     * is interpreted as the key, and the second substring after an arbitrary
     * amount of ' ' is interpreted as the value. Following substrings are
     * ignored in that line. Still, it is recommended to start comments with a
     * '#' for better readability.
     *
     * Example `config.txt`:
     * @code
     * # This is a comment
     * train_data_path = /path/to/train/data # Path to training data
     * train_target_path = /path/to/train/target # Path to training target
     * test_data_path = /path/to/test/data # Path to test data
     * test_target_path = /path/to/test/target # Path to test target
     * output_path = /path/to/output # Path to output
     * batchsize = 32 # Batch size for training
     * epochs = 10 # Number of epochs
     * learning_rate = 0.01 # Learning rate
     * @endcode
     *
     * @note The available keys have the same name as the public member
     * variables of this class.
     *
     * @attention When adding new parameters, the user needs to add them to the
     * public section and also use the `CONFIG_PARSE_ENTRY` macro to add them to
     * the `dispatch_table` in the private section of the class.
     *
     * @tparam data_type The data type for the configuration parameters.
     */
    template <core::arithmetic_type data_type>
    class Config {
      public:
        // general config values
        std::string train_data_path;
        std::string train_target_path;
        std::string test_data_path;
        std::string test_target_path;
        std::string output_path;
        core::index_type batchsize = 1;
        core::index_type epochs = std::numeric_limits<core::index_type>::max();
        core::index_type iterations =
            std::numeric_limits<core::index_type>::max();
        data_type learning_rate = 1e-3;
        bool benchmark = false;
        core::index_type warmup_steps = 0;
        bool tf32 = true;
        bool progress = true;

        // additional parameters for MNIST:
        core::label_type num_classes = 10;
        core::index_type input_1D = 100;
        core::index_type hidden_dim = 1000;

        // additional parameters for MNIST_Exteded:
        data_type dropout = 0.5;

        // additional parameters for Transformer:
        core::index_type sequence_length = 1;
        core::index_type embedding_dim = 128;
        core::index_type num_encoder_layers = 3;
        core::index_type num_decoder_layers = 3;
        core::index_type num_heads = 8;
        core::label_type num_embeddings = 128;
        core::index_type max_batchsize = 1000;
        core::index_type ignore_index =
            std::numeric_limits<core::index_type>::max();
        core::index_type noam_warmup_steps = 4000;

        // additional multi-purpose parameters for Benchmarking
        core::index_type L = 1;
        core::index_type M = 1;
        core::index_type N = 1;

      private:
        /**
         * @brief This macro creates the expected dispatch table entry from a
         * given variable name.
         *
         * This macro creates the key for each variable by converting the
         * name to a string. Additionally, it creates a lambda function that
         * writes a given value into the specified variable.
         *
         * For example, `CONFIG_PARSE_ENTRY(x)` will be converted to:
         * `{"x", [](Config<data_type> *config, const std::string& value){
         * std::istringstream(value) >> config->x;}}`
         */
        #define CONFIG_PARSE_ENTRY(member) {                                   \
            #member,                                                           \
            [](Config <data_type> * config, const std::string &value) {        \
                std::istringstream(value) >> config->member;                   \
            }                                                                  \
        }

        /**
         * @brief The dispatch_table for parsing the key value pairs.
         */
        std::unordered_map<
            std::string,
            std::function<void(Config<data_type> *, const std::string &)>>
            dispatch_table{
                // general config values
                CONFIG_PARSE_ENTRY(train_data_path),
                CONFIG_PARSE_ENTRY(train_target_path),
                CONFIG_PARSE_ENTRY(test_data_path),
                CONFIG_PARSE_ENTRY(test_target_path),
                CONFIG_PARSE_ENTRY(output_path), CONFIG_PARSE_ENTRY(batchsize),
                CONFIG_PARSE_ENTRY(epochs), CONFIG_PARSE_ENTRY(iterations),
                CONFIG_PARSE_ENTRY(learning_rate),
                CONFIG_PARSE_ENTRY(benchmark), CONFIG_PARSE_ENTRY(warmup_steps),
                CONFIG_PARSE_ENTRY(tf32), CONFIG_PARSE_ENTRY(progress),

                // additional parameters for MNIST:
                CONFIG_PARSE_ENTRY(input_1D), CONFIG_PARSE_ENTRY(hidden_dim),

                // additional parameters for MNIST_Exteded:
                CONFIG_PARSE_ENTRY(dropout),

                // additional parameters for Transformer:
                CONFIG_PARSE_ENTRY(sequence_length),
                CONFIG_PARSE_ENTRY(embedding_dim),
                CONFIG_PARSE_ENTRY(num_encoder_layers),
                CONFIG_PARSE_ENTRY(num_decoder_layers),
                CONFIG_PARSE_ENTRY(num_heads),
                CONFIG_PARSE_ENTRY(num_embeddings),
                CONFIG_PARSE_ENTRY(max_batchsize),
                CONFIG_PARSE_ENTRY(ignore_index),

                // additional multi-purpose parameters for Benchmarking
                CONFIG_PARSE_ENTRY(L),
                CONFIG_PARSE_ENTRY(M),
                CONFIG_PARSE_ENTRY(N)
            };

      public:
        Config() = default;

        /**
         * @brief Constructs a new Config object.
         *
         * @param argc The number of command line arguments.
         * @param argv The command line arguments. The first argument will be
         * ignored because it is assumed to be the program name, similar to the
         * `argv` parameter in the `main` function.
         */
        Config(const int argc, const char *const argv[]) {
            std::vector<std::string> args{argv + 1, argv + argc};

            if (args.empty())
                throw std::invalid_argument(
                    "You need to provide at least one config file as a command "
                    "line argument!");

            for (const auto &arg : args)
                if (!contains(arg, '=')) parse_config_file(arg);

            parse(args);
        }

        /**
         * @brief Constructs a new Config object.
         *
         * @param path The path to the config file.
         */
        Config(const std::string &path) { parse_config_file(path); }

        /**
         * @brief Returns the memory footprint of the object on GPU.
         */
        const core::Mem_Size mem_size(const bool /* unused */ = false) const {
            return {0, 0};
        }

      private:
        /**
         * @brief Checks if a string contains a specific character.
         *
         * @param str The string to check.
         * @param c The character to look for.
         * @return true If the character is found.
         * @return false If the character is not found.
         */
        bool contains(const std::string &str, const char c) {
            return str.find(c) != std::string::npos;
        }

        /**
         * @brief Parses a configuration file.
         *
         * @param path The path to the configuration file.
         */
        void parse_config_file(const std::string &path) {
            std::vector<std::string> args;

            std::ifstream file(path);
            if (!file.is_open())
                throw std::runtime_error("Could not open config file: " + path);

            std::string line;
            while (std::getline(file, line)) args.push_back(line);

            parse(args);
        }

        /**
         * @brief Parses a vector of configuration arguments.
         *
         * @param args The vector of configuration arguments.
         */
        void parse(const std::vector<std::string> &args) {
            for (const auto &arg : args) {
                if (!contains(arg, '=')) continue;
                if (arg[0] == '#') continue;

                std::string temp = arg;
                std::replace(temp.begin(), temp.end(), '=', ' ');

                std::string key, value;
                std::istringstream(temp) >> key >> value;

                auto it = dispatch_table.find(key);
                if (it != dispatch_table.end()) it->second(this, value);
            }
        }
    };

}  // namespace utils
