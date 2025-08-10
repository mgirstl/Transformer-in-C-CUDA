/**
 * @file 03_losses.cu
 * @brief Tests all the classes in the namespace `losses`.
 */

#include <cmath>
#include <limits>

#include "../src/core.cuh"
#include "../src/losses.cuh"
#include "../src/test.cuh"

/**
 * @brief This function tests the CrossEntropy loss using random data and
 * special cases.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_crossentropy(test::Test &test) {
    test.component("CrossEntropy");

    core::Device device;

    // random input
    std::string path = test.data_path("losses/crossentropy");
    core::Device_Tensor<core::data_type> input(path + "_input");

    losses::CrossEntropy<core::data_type, core::label_type> crossentropy(
        device, input.shape_vec(core::NO_BATCHSIZE));

    core::Device_Tensor<core::label_type> target(path + "_target");
    core::Host_Tensor<core::data_type> output(path + "_output");
    test.check(test::compare(crossentropy.forward(input, target), output),
               "forward correct");

    core::Host_Tensor<core::data_type> input_gradient(path + "_input_gradient");
    test.check(test::compare(crossentropy.backward(), input_gradient),
               "backward correct");

    // special case (zero input)
    std::string path_special_cases =
        test.data_path("losses/crossentropy_special_cases");
    core::Device_Tensor<core::data_type> input_special_cases(
        path_special_cases + "_input");

    losses::CrossEntropy<core::data_type, core::label_type>
        crossentropy_special_cases(
            device, input_special_cases.shape_vec(core::NO_BATCHSIZE));

    core::Device_Tensor<core::label_type> target_special_cases(
        path_special_cases + "_target");

    auto output_special_case =
        core::Host_Tensor(crossentropy_special_cases.forward(
            input_special_cases, target_special_cases));
    auto input_gradient_special_case =
        core::Host_Tensor(crossentropy_special_cases.backward());

    core::data_type val = 0;
    for (core::index_type idx = 0; idx < output_special_case.size(); ++idx)
        val += output_special_case[idx] / output_special_case.size();
    test.check(val > -std::log(std::numeric_limits<core::data_type>::epsilon()),
               "forward correct (zero input)");

    val = 0;
    for (core::index_type idx = 0; idx < input_gradient_special_case.size();
         ++idx)
        val += input_gradient_special_case[idx];
    test.check(val < -1 / std::numeric_limits<core::data_type>::epsilon(),
               "backward correct (zero input)");

    // random 3D input
    std::string path_3D = test.data_path("losses/crossentropy_3D");
    core::Device_Tensor<core::data_type> input_3D(path_3D + "_input");

    losses::CrossEntropy<core::data_type, core::label_type> crossentropy_3D(
        device, input_3D.shape_vec(core::NO_BATCHSIZE), 0);

    core::Device_Tensor<core::label_type> target_3D(path_3D + "_target");
    core::Host_Tensor<core::data_type> output_3D(path_3D + "_output");
    test.check(
        test::compare(crossentropy_3D.forward(input_3D, target_3D), output_3D),
        "forward correct 3D");

    core::Host_Tensor<core::data_type> input_gradient_3D(path_3D +
                                                         "_input_gradient");
    test.check(test::compare(crossentropy_3D.backward(), input_gradient_3D),
               "backward correct 3D");
}

/**
 * @brief Launches all tests defined in this file.
 *
 * This function initializes the test framework using command line arguments
 * and runs a series of tests for the loss functions defined in the `losses`
 * namespace.
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
    test.series("Losses");
    test_crossentropy(test);
    return test.eval();
}
