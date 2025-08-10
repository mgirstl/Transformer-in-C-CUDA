/**
 * @file 08_models_double.cu
 * @brief Tests all the classes in the `models` namespace with `double` as
 * tensor element type instead of `float`.
 *
 * @note This is a separate file from `07_models.cu` because compiling both in
 * the same file led to compiler issues with the `NVC++` compiler
 * (version 23.7-0). The error encountered was:
 * @code
 * nvvmCompileProgram error 9: NVVM_ERROR_COMPILATION.
 * Error: /tmp/2088250.alex/nvacc97KL3ltXLJPr.gpu (15337, 49): parse
 * '@shared_max' defined with type '[0 x float] addrspace(3)*'
 * NVC++-F-0155-Compiler failed to translate accelerator region
 * @endcode
 */

#include "../src/core.cuh"
#include "../src/models.cuh"
#include "../src/optimizers.cuh"
#include "../src/test.cuh"
#include "../src/utils.cuh"

/**
 * @brief This function tests the MNIST class by overfitting the model.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_mnist(test::Test &test) {
    test.component("MNIST");
    std::string path = test.data_path("models/mnist");

    core::Device device;

    test.comment("Build...");
    utils::Config<double> config{path + "_config"};
    models::MNIST<double, core::label_type, optimizers::SGD> model{device,
                                                                   config};
    test.comment("\n" + model.info());

    test.comment("Evaluation before training...");
    double base_accuracy = model.accuracy(false, true);

    test.comment("Training...");
    auto iterations = model.train(false, true);

    test.comment("Evaluation...");
    double accuracy = model.accuracy(false, true);
    test.check(base_accuracy < 0.8 && accuracy > 0.8,
               "Overfit accuracy correct");

    // test prediction function and different batchsize
    core::Device_Tensor<double> data(path + "_data");
    core::Host_Tensor<core::index_type> target(path + "_target");
    auto prediction = core::Host_Tensor(model.predict(data));

    core::index_type sum = 0;
    for (core::index_type idx = 0; idx < prediction.size(); ++idx)
        sum += prediction[idx] == target[idx];

    test.check(1.0 * sum / prediction.size() > 0.8,
               "Overfit prediction correct (small batchsize)");

    data.load(path + "_data_large");
    target.load(path + "_target_large");
    prediction = core::Host_Tensor(model.predict(data));

    sum = 0;
    for (core::index_type idx = 0; idx < prediction.size(); ++idx)
        sum += prediction[idx] == target[idx];
    test.check(1.0 * sum / prediction.size() > 0.8,
               "Overfit prediction correct (large batchsize)");

    // test save/load
    model.save(test.temporary_path("mnist"));
    models::MNIST<double, core::label_type, optimizers::SGD> model_2{device,
                                                                     config};
    double base_accuracy_2 = model_2.accuracy();
    model_2.load(test.temporary_path("mnist"));
    double accuracy_2 = model_2.accuracy();
    test.check(base_accuracy_2 < 0.8 && test::compare(accuracy, accuracy_2),
               "Overfit accuracy after save/load correct");
}

/**
 * @brief This function tests the MNIST_Extended class by overfitting the model.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_mnist_extended(test::Test &test) {
    test.component("MNIST Extended");
    std::string path = test.data_path("models/mnist");
    core::Device device;

    test.comment("Build...");
    utils::Config<double> config{path + "_config"};
    models::MNIST_Extended<double, core::label_type, optimizers::Adam> model{
        device, config};
    test.comment("\n" + model.info());

    test.comment("Evaluation before training...");
    double base_accuracy = model.accuracy(false, true);

    test.comment("Training...");
    auto iterations = model.train(false, true);

    test.comment("Evaluation...");
    double accuracy = model.accuracy(false, true);
    test.check(base_accuracy < 0.8 && accuracy > 0.8,
               "Overfit accuracy correct");

    // test save/load
    model.save(test.temporary_path("mnist_extended"));
    models::MNIST_Extended<double, core::label_type, optimizers::Adam> model_2{
        device, config};
    double base_accuracy_2 = model_2.accuracy();
    model_2.load(test.temporary_path("mnist_extended"));
    double accuracy_2 = model_2.accuracy();
    test.check(base_accuracy_2 < 0.8 && test::compare(accuracy, accuracy_2),
               "Overfit accuracy after save/load correct");
}

/**
 * @brief This function tests the Transformer class by overfitting the model.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_transformer(test::Test &test) {
    test.component("Transformer");
    std::string path = test.data_path("models/transformer");
    core::Device device;

    test.comment("Build...");
    utils::Config<double> config{path + "_config"};
    models::Transformer<core::label_type, double, optimizers::Adam> model{
        device, config};
    test.comment("\n" + model.info());

    test.comment("Evaluation before training...");
    double base_accuracy = model.accuracy(false, true);

    test.comment("Training...");
    auto iterations = model.train(false, true);

    test.comment("Evaluation...");
    double accuracy = model.accuracy(false, true);
    test.check(base_accuracy < 0.8 && accuracy > 0.8,
               "Overfit accuracy correct");

    // test save/load
    model.save(test.temporary_path("transformer"));
    models::Transformer<core::label_type, double, optimizers::Adam> model_2{
        device, config};
    double base_accuracy_2 = model_2.accuracy();
    model_2.load(test.temporary_path("transformer"));
    double accuracy_2 = model_2.accuracy();
    test.check(base_accuracy_2 < 0.8 && test::compare(accuracy, accuracy_2),
               "Overfit accuracy after save/load correct");
}

/**
 * @brief Launches all tests defined in this file.
 *
 * This function initializes the test framework using command line arguments
 * and runs a series of tests for the models defined in the `models` namespace.
 *
 * @param argc The number of command line arguments.
 * @param argv The command line arguments. The first argument will be ignored
 * because it is assumed to be the program name. The second should be the path
 * to the data directory for the tests, and the third should be a path to a
 * temporary directory for use by tests.
 * @return The number of tests that failed.
 */
int main(int argc, char *argv[]) {
    test::Test test(argc, argv);
    test.series("Models (double)");
    test_mnist(test);
    test_mnist_extended(test);
    test_transformer(test);
    return test.eval();
}
