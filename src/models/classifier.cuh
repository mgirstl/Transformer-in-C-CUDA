/**
 * @file classifier.cuh
 * @brief Implements the interface for classifiers.
 *
 * This file is part of the `models` namespace, which implements neural network
 * models.
 */

#pragma once

#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../core.cuh"
#include "../losses/crossentropy.cuh"
#include "../optimizers/optimizer.cuh"
#include "../utils.cuh"

/**
 * @namespace models
 * @brief Implements neural network models.
 *
 * @internal
 * This specific file defines the Classifier interface.
 * @endinternal
 */
namespace models {

    /**
     * @interface Classifier
     * @brief Interface for classifiers.
     *
     * @tparam input_type The type of the input tensor.
     * @tparam data_type The type used for calculations inside of the network.
     * @tparam label_type The type of the labels of the network.
     * @tparam optimizer_type The optimizer used for the parameter updates.
     */
    template <core::arithmetic_type input_type, core::float_type data_type,
              core::integer_type label_type,
              template <typename> class optimizer_type>
    class Classifier {
        static_assert(
            optimizers::optimizer_type<optimizer_type<data_type>, data_type>,
            "optimizer_type must satisfy the optimizers::optimizer_type "
            "concept!");

      protected:
        const core::Device &device;
        const utils::Config<data_type> &config;
        core::index_type ignore_index;
        utils::Indicator<label_type> indicator;
        utils::Argmax<data_type, label_type> argmax;
        losses::CrossEntropy<data_type, label_type> crossentropy;
        std::vector<data_type> loss_history;

      public:
        /**
         * @brief Construct a new Classifier object.
         *
         * @param device The device object that calculates the recommended
         * number of threads and blocks.
         * @param config The config used for creating the Classifier. It saves
         * among others the path to the training and evaluation data.
         * @param num_classes The number of classes the classifier approximates.
         * @param sequence_length The size of additional dimensions in the
         * output of the classifier, i.e., if the output of the classifier has
         * the shape `[batchsize, *, num_classes]`, then the `sequence_length`
         * is the total size of all the additional dimensions '*'.
         * @param ignore_index Specifies a target value that is ignored and does
         * not contribute to the loss and/or accuracy.
         */
        Classifier(core::Device &device, const utils::Config<data_type> &config,
                   const label_type num_classes,
                   const core::index_type sequence_length = 1,
                   const label_type ignore_index =
                       std::numeric_limits<label_type>::max())
            : device{device},
              config{config},
              ignore_index{ignore_index},
              indicator{device, {sequence_length}},
              argmax{device, {sequence_length, num_classes}, ignore_index},
              crossentropy{
                  device, {sequence_length, num_classes}, ignore_index} {}

        /**
         * @brief Executes a single training step.
         *
         * @param data The data to process during this training step.
         * @param target The target values belonging to the data.
         * @param stream The CUDA stream used for launching the kernels.
         * @return The average loss of the training step.
         */
        data_type training_step(
            const core::Device_Tensor<input_type> &data,
            const core::Device_Tensor<label_type> &target,
            const cudaStream_t stream = core::default_stream) {
            // create core::Stream and let them wait for the provided stream
            core::Stream copy_stream;
            core::Stream calculation_stream;
            copy_stream.wait(stream);
            calculation_stream.wait(stream);

            // forward pass
            const auto &x1 = forward(data, calculation_stream);
            const auto &_loss =
                crossentropy.forward(x1, target, calculation_stream);

            // copy loss to host
            copy_stream.wait(calculation_stream);
            const auto loss = core::Host_Tensor(_loss, copy_stream);

            // backward pass
            const auto &x2 = crossentropy.backward(calculation_stream);
            backward(x2, calculation_stream);

            // calculate average loss
            data_type sum = 0;
            for (core::index_type idx = 0; idx < loss.size(); ++idx)
                sum += loss[idx];

            // let the provided stream wait for the "local" core::Streams
            copy_stream.make_wait(stream);
            calculation_stream.make_wait(stream);

            return sum / loss.size();
        }

        /**
         * @brief Trains a neural network.
         *
         * @note If `config.warmup_steps > 0`, then some additional warmup steps
         * are performed before the "main training" starts. This means the
         * number of actual performed training steps is `config.warmup_steps`
         * larger than given by `config.epochs` or `config.iterations`.
         *
         * @param progress If true, then occasionally additional info, e.g.,
         * the current epoch, loss and estimated time of arrival is printed.
         * @param statistics If true, then print additional statistics after
         * the training.
         * @return The actual number of iterations the network has been trained
         * for including warmup steps.
         */
        core::index_type train(const bool progress = false,
                               const bool statistics = false) {
            // activate training mode
            _train();

            // load data
            utils::DataLoader<input_type, label_type> dataloader{
                device,
                config.train_data_path,
                config.train_target_path,
                config.batchsize,
                true,
                config.epochs,
                config.iterations};

            // warmup
            for (core::index_type step = 0; step < config.warmup_steps;
                 ++step) {
                dataloader.next();
                training_step(dataloader.data(), dataloader.target());
            }
            dataloader.reset();

            // calculate rate in which the progress will be printed
            const auto max_iterations = calc_max_iterations(
                dataloader.num_samples(), config.epochs, config.iterations);
            const auto progress_rate = calc_progress_rate(max_iterations);

            // start the actual training
            core::checkCuda(cudaDeviceSynchronize());
            const auto start = std::chrono::system_clock::now();

            while (dataloader.next()) {
                const auto loss =
                    training_step(dataloader.data(), dataloader.target());
                loss_history.push_back(loss);

                if (progress &&
                    (dataloader.iteration() + 1) % progress_rate == 0) {
                    print_progress(start, dataloader.epoch() + 1,
                                   dataloader.iteration() + 1, max_iterations,
                                   loss);
                }
            }

            core::checkCuda(cudaDeviceSynchronize());
            const auto end = std::chrono::system_clock::now();

            // print statistics
            if (statistics) {
                const auto duration =
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        end - start);
                std::cout << ">>> Training Statistics:" << std::endl;
                std::cout << "Total Runtime [ms]: " << duration.count() / 1000.
                          << std::endl;
                std::cout << "Runtime per Iteration [ms]: "
                          << duration.count() / (1000. * dataloader.iteration())
                          << std::endl;
                std::cout << "Final Loss: " << loss_history.back() << std::endl;
                std::cout << "Epochs: " << dataloader.epoch() << std::endl;
                std::cout << "Iterations: " << dataloader.iteration()
                          << std::endl;
                std::cout << "Parameters: " << parameters() << std::endl;
                std::cout << "###############################" << std::endl;
            }

            return dataloader.iteration() + config.warmup_steps;
        }

        /**
         * @brief Calculates the accuracy of the network.
         *
         * @param progress If true, then occasionally additional info, e.g.,
         * the current epoch, and estimated time of arrival is printed.
         * @param statistics If true, then print additional statistics after
         * the training.
         * @return The accuracy.
         */
        data_type accuracy(const bool progress = false,
                           const bool statistics = false) {
            // activate evaluation mode
            _eval();

            // load data
            utils::DataLoader<input_type, label_type> dataloader{
                device,
                config.test_data_path,
                config.test_target_path,
                config.batchsize,
                false,
                1};

            // calculate rate in which the progress will be printed
            const auto max_iterations =
                calc_max_iterations(dataloader.num_samples(), 1);
            const auto progress_rate = calc_progress_rate(max_iterations);

            // start the evaluation
            core::checkCuda(cudaDeviceSynchronize());
            const auto start = std::chrono::system_clock::now();

            core::index_type correct = 0;
            core::index_type total = 0;

            while (dataloader.next()) {
                const auto &prediction = predict(dataloader.data());
                const auto &_target = dataloader.target();
                const auto &_output = indicator.apply(prediction, _target);

                const auto output = core::Host_Tensor(_output);
                const auto target = core::Host_Tensor(_target);

                for (core::index_type idx = 0; idx < config.batchsize; ++idx) {
                    if (dataloader.iteration() * config.batchsize + idx >=
                        dataloader.num_samples())
                        break;
                    if (target[idx] == ignore_index) continue;
                    correct += output[idx];
                    total += 1;
                }

                if (progress &&
                    (dataloader.iteration() + 1) % progress_rate == 0) {
                    print_progress(start, dataloader.epoch() + 1,
                                   dataloader.iteration() + 1, max_iterations);
                }
            }

            core::checkCuda(cudaDeviceSynchronize());
            const auto end = std::chrono::system_clock::now();

            // print statistics
            if (statistics) {
                const auto duration =
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        end - start);
                std::cout << ">>> Evaluation Statistics:" << std::endl;
                std::cout << "Total Runtime [ms]: " << duration.count() / 1000.
                          << std::endl;
                std::cout << "Runtime per Iteration [ms]: "
                          << duration.count() / (1000. * dataloader.iteration())
                          << std::endl;
                std::cout << "Accuracy: " << correct / (1.0 * total)
                          << std::endl;
                std::cout << "###############################" << std::endl;
            }

            return correct / (1.0 * total);
        }

        /**
         * @brief Compute the prediction of the classifier.
         *
         * @param input The input tensor.
         * @param stream The CUDA stream to use for computation.
         * @return The output tensor.
         */
        const core::Device_Tensor<label_type> &predict(
            const core::Device_Tensor<input_type> &input,
            const cudaStream_t stream = core::default_stream) {
            _eval();
            const auto &x = forward(input, stream);
            return argmax.apply(x, stream);
        }

        /**
         * @brief Get the Indicator function.
         */
        const utils::Indicator<label_type> &get_indicator() const {
            return indicator;
        }

        /**
         * @brief Get the Argmax function.
         */
        const utils::Argmax<data_type, label_type> &get_argmax() const {
            return argmax;
        }

        /**
         * @brief Get the CrossEntropy function.
         */
        const losses::CrossEntropy<data_type, label_type> &get_crossentropy()
            const {
            return crossentropy;
        }

        /**
         * @brief Get the loss history.
         */
        const std::vector<data_type> &get_loss_history() const {
            return loss_history;
        }

        /**
         * @brief Saves the parameters of the classifier to a file.
         *
         * @param path The base path to write the parameters to.
         *
         * @note It is assumed that the filetype is "tensor".
         */
        void save(const std::string &path) { _save(path, "tensor"); }

        /**
         * @brief Saves the parameters of the classifier to a file.
         *
         * @param path The base path to write the parameters to.
         * @param filetype The filetype of the saved files.
         */
        void save(const std::string &path, const std::string &filetype) {
            _save(path, filetype);
        }

        /**
         * @brief Loads the parameters of the classifier from a file.
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
         * @brief Loads the parameters of the classifier from a file.
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
         * @brief Loads the parameters of the classifier from a file.
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
         * @brief Loads the parameters of the classifier from a file.
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
         * @brief Returns the number of trainable parameters of the classifier.
         */
        virtual core::index_type parameters() const = 0;

        /**
         * @brief Returns the memory footprint of the object on GPU.
         *
         * @note This function should be overridden by derived classes if
         * the derived class allocates memory on the GPU.
         */
        virtual const core::Mem_Size mem_size(
            const bool /* unused */ = false) const {
            core::Mem_Size size;
            size += indicator.mem_size();
            size += argmax.mem_size();
            size += crossentropy.mem_size();
            return size;
        }

        /**
         * @brief Returns information about the classifier.
         */
        virtual const std::string info() const = 0;

        /**
         * @brief Destroy the Classifier object.
         *
         * Ensures correct behavior when deleting derived classes.
         */
        virtual ~Classifier() {}

      protected:
        /**
         * @brief Compute the forward pass of the classifier.
         *
         * @param input The input tensor.
         * @param stream The CUDA stream to use for computation.
         * @return The output tensor.
         */
        virtual const core::Device_Tensor<data_type> &forward(
            const core::Device_Tensor<input_type> &input,
            const cudaStream_t stream) = 0;

        /**
         * @brief Compute the backward pass of the classifier.
         *
         * @param error The error tensor from the loss function.
         * @param stream The CUDA stream to use for computation.
         */
        virtual void backward(const core::Device_Tensor<data_type> &error,
                              const cudaStream_t stream) = 0;

        /**
         * @brief Sets the classifier into training mode.
         *
         * This changes the calculation in the forward pass for some layers,
         * e.g., Dropout. Moreover, in training mode, the parameter update
         * step will be performed when calling the backward pass of the layer.
         *
         * In general, layers are initialized in training mode.
         */
        virtual void _train() = 0;

        /**
         * @brief Sets the classifier into evaluation mode.
         *
         * This changes the calculation in the forward pass for some layers,
         * e.g., Dropout. Moreover, in evaluation mode, no parameter update
         * is performed in the backward pass.
         *
         * In general, layers are initialized in training mode.
         */
        virtual void _eval() = 0;

        /**
         * @brief Saves the parameters of the classifier to a file.
         *
         * @param path The base path to write the parameters to.
         * @param filetype The filetype of the saved files.
         */
        virtual void _save(const std::string &path,
                           const std::string &filetype) = 0;

        /**
         * @brief Loads the parameters of the classifier from a file.
         *
         * @param path The base path to read the parameters from.
         * @param filetype The filetype of the saved files.
         * @param stream The CUDA stream used for copying from the host.
         */
        virtual void _load(const std::string &path, const std::string &filetype,
                           const cudaStream_t &stream) = 0;

      private:
        /**
         * @brief Calculates the maximum number of iterations.
         *
         * Since the number of iterations is limited by either the maximum
         * number of iterations or the maximum number of epochs, the actual
         * maximum number of iterations can be smaller than the give maximum
         * number of iterations.
         *
         * @param num_samples The total number of samples to process in one
         * epoch.
         * @param epochs The maximum number of epochs.
         * @param iterations The maximum number of iterations.
         * @return The total maximum number of iterations.
         */
        core::index_type calc_max_iterations(
            const core::index_type num_samples, const core::index_type epochs,
            const core::index_type iterations =
                std::numeric_limits<core::index_type>::max()) const {
            return min(epochs * (num_samples + config.batchsize - 1) /
                           config.batchsize,
                       iterations);
        }

        /**
         * @brief Calculates how often the progress should be printed.
         *
         * @param max_iterations The maximum number of iterations.
         * @return The progress rate.
         */
        core::index_type calc_progress_rate(
            const core::index_type max_iterations) const {
            core::index_type progress_rate = 100;
            if (max_iterations / 10 < progress_rate && max_iterations / 10 > 0)
                progress_rate = max_iterations / 10;
            return progress_rate;
        }

        /**
         * @brief Prints the current progress.
         *
         * @param start The start time of the calculation.
         * @param epoch The number of elapsed epochs.
         * @param iteration The number of elapsed iterations.
         * @param max_iterations The maximum number of iterations.
         * @param loss If `NAN`, then the loss will be omitted from the output.
         */
        void print_progress(
            const std::chrono::time_point<std::chrono::system_clock> &start,
            const core::index_type epoch, const core::index_type iteration,
            const core::index_type max_iterations,
            const data_type loss = NAN) const {
            const auto current_time = std::chrono::system_clock::now();
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    current_time - start)
                    .count();
            const auto eta = elapsed * (max_iterations - iteration) / iteration;
            const auto eta_minutes = eta / 60000;
            const auto eta_seconds = (eta % 60000) / 1000;

            auto original_flags = std::cout.flags();
            auto original_precision = std::cout.precision();
            std::cout << std::fixed << std::setprecision(4);

            if (std::isnan(loss))
                std::cout << "Epoch: " << std::right << std::setw(5) << epoch
                          << ", Iteration: " << std::right << std::setw(5)
                          << iteration << ", ETA: " << std::right
                          << std::setw(3) << eta_minutes << " min "
                          << std::right << std::setw(2) << eta_seconds << " sec"
                          << std::endl;
            else
                std::cout << "Epoch: " << std::right << std::setw(5) << epoch
                          << ", Iteration: " << std::right << std::setw(5)
                          << iteration << ", Loss: " << std::right
                          << std::setw(5) << loss << ", ETA: " << std::right
                          << std::setw(3) << eta_minutes << " min "
                          << std::right << std::setw(2) << eta_seconds << " sec"
                          << std::endl;

            std::cout.flags(original_flags);
            std::cout.precision(original_precision);
            std::cout << std::defaultfloat;
        }
    };

}  // namespace models
