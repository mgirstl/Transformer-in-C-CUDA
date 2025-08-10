/**
 * @file transformer.cu
 * @brief This file executes the Transformer training and evaluation.
 */

#include <iostream>

#include "../src/core.cuh"
#include "../src/models/transformer.cuh"
#include "../src/optimizers.cuh"
#include "../src/utils/config.cuh"

/**
 * @brief Main function to train the Transformer model.
 *
 * @param argc The number of command line arguments.
 * @param argv The command line arguments. The first argument will be ignored
 * because it is assumed to be the program name. The second should be the path
 * to a config file. For more information on the config file, see
 * `utils::Config`.
 */
int main(int argc, char *argv[]) {
    utils::Config<core::data_type> config(argc, argv);

    core::Device device;

    if (config.tf32)
        core::checkCuda(cublasSetMathMode(device.cublas_handle(),
                                          CUBLAS_TF32_TENSOR_OP_MATH));

    std::cout << "Build..." << std::endl;
    models::Transformer<core::label_type, core::data_type, optimizers::Adam>
        model{device,
              config};  // alternative: optimizers::Adam_with_Noam_Scheduler
    std::cout << model.info() << std::endl;

    std::cout << "Training..." << std::endl;
    model.train(config.progress, true);

    std::cout << "Evaluation..." << std::endl;
    if (config.benchmark)
        std::cout << "Skipping Evaluation!" << std::endl;
    else
        model.accuracy(false, true);

    std::cout << ">>> Memory Statistics:" << std::endl;
    std::cout << "Calculated Usage [MB]: "
              << (model.mem_size().fixed_size +
                  model.mem_size().variable_size * config.batchsize) /
                     (1024. * 1024.)
              << std::endl;
    std::cout << "Measured Usage [MB]: "
              << utils::get_memory_usage() / (1024. * 1024.) << std::endl;
    std::cout << "###############################" << std::endl;
}
