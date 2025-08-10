/**
 * @file 02_activations.cu
 * @brief Tests all the classes in the namespace `activations`.
 */
#include <cmath>

#include "../src/activations.cuh"
#include "../src/core.cuh"
#include "../src/test.cuh"

/**
 * @brief This function tests the Identity function using random data.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_identity(test::Test &test) {
    test.component("Identity");

    core::Host_Tensor<int> host_tensor({3, 4, 5});
    for (core::index_type idx = 0; idx < host_tensor.size(); ++idx)
        host_tensor[idx] = idx;
    core::Device_Tensor<int> device_tensor(host_tensor);

    activations::Identity<int> identity;

    test.check(test::compare(identity.forward(device_tensor), host_tensor),
               "forward correct");
    test.check(test::compare(identity.backward(device_tensor), host_tensor),
               "backward correct");
}

/**
 * @brief This function tests the ReLU function using random data and special
 * cases.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_relu(test::Test &test) {
    test.component("ReLU");
    std::string path = test.data_path("activations/relu");

    core::Device device;

    // test on random dataset
    core::Device_Tensor<core::data_type> input(path + "_input");

    activations::ReLU<core::data_type> relu(
        device, input.shape_vec(core::NO_BATCHSIZE));

    core::Host_Tensor<core::data_type> output(path + "_output");
    test.check(test::compare(relu.forward(input), output), "forward correct");

    core::Device_Tensor<core::data_type> error(path + "_error");
    core::Host_Tensor<core::data_type> input_gradient(path + "_input_gradient");
    test.check(test::compare(relu.backward(error), input_gradient),
               "backward correct");

    // second run
    test.check(test::compare(relu.forward(input), output),
               "forward correct (second run)");
    test.check(test::compare(relu.backward(error), input_gradient),
               "backward correct (second run)");

    // special cases
    activations::ReLU<core::data_type> relu_special_case(device, {});
    core::Host_Tensor<core::data_type> input_special_case({3});
    input_special_case[0] = -1e100;
    input_special_case[1] = 0;
    input_special_case[2] = 1e100;

    core::Host_Tensor<core::data_type> output_special_case({3});
    output_special_case[0] = 0;
    output_special_case[1] = 0;
    output_special_case[2] = 1e100;

    auto result =
        relu_special_case.forward(core::Device_Tensor(input_special_case));
    test.check(test::compare(result, output_special_case),
               "special case forward correct");

    core::Host_Tensor<core::data_type> error_special_case({3});
    error_special_case[0] = 1e100;
    error_special_case[1] = 1e100;
    error_special_case[2] = 1e100;

    core::Host_Tensor<core::data_type> input_gradient_special_case({3});
    input_gradient_special_case[0] = 0;
    input_gradient_special_case[1] = 0;
    input_gradient_special_case[2] = 1e100;

    result =
        relu_special_case.backward(core::Device_Tensor(error_special_case));
    test.check(test::compare(result, input_gradient_special_case),
               "special case backward correct");
}

/**
 * @brief This function tests the Softmax function using random data and special
 * cases.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_softmax(test::Test &test) {
    test.component("Softmax");

    core::Device device;

    // random 2D dataset
    std::string path_2D = test.data_path("activations/softmax_2D");
    core::Device_Tensor<core::data_type> input_2D(path_2D + "_input");

    activations::Softmax<core::data_type> softmax_2D(
        device, input_2D.shape_vec(core::NO_BATCHSIZE));

    core::Host_Tensor<core::data_type> output_2D(path_2D + "_output");
    test.check(test::compare(softmax_2D.forward(input_2D), output_2D),
               "forward correct (2D)");

    core::Device_Tensor<core::data_type> error_2D(path_2D + "_error");
    core::Host_Tensor<core::data_type> input_gradient_2D(path_2D +
                                                         "_input_gradient");
    test.check(test::compare(softmax_2D.backward(error_2D), input_gradient_2D),
               "backward correct (2D)");

    // second run
    test.check(test::compare(softmax_2D.forward(input_2D), output_2D),
               "forward correct (2D, second run)");
    test.check(test::compare(softmax_2D.backward(error_2D), input_gradient_2D),
               "backward correct (2D, second run)");

    // random 10D dataset
    std::string path_10D = test.data_path("activations/softmax_10D");
    core::Device_Tensor<core::data_type> input_10D(path_10D + "_input");

    activations::Softmax<core::data_type> softmax_10D(
        device, input_10D.shape_vec(core::NO_BATCHSIZE));

    core::Host_Tensor<core::data_type> output_10D(path_10D + "_output");
    test.check(test::compare(softmax_10D.forward(input_10D), output_10D),
               "forward correct (10D)");

    core::Device_Tensor<core::data_type> error_10D(path_10D + "_error");
    core::Host_Tensor<core::data_type> input_gradient_10D(path_10D +
                                                          "_input_gradient");
    test.check(
        test::compare(softmax_10D.backward(error_10D), input_gradient_10D),
        "backward correct (10D)");

    // special cases
    std::string path_special_cases =
        test.data_path("activations/softmax_special_cases");
    core::Device_Tensor<core::data_type> input_special_cases(
        path_special_cases + "_input");

    activations::Softmax<core::data_type> softmax_special_cases(
        device, input_special_cases.shape_vec(core::NO_BATCHSIZE));

    core::Host_Tensor<core::data_type> output_special_cases(path_special_cases +
                                                            "_output");
    test.check(test::compare(softmax_special_cases.forward(input_special_cases),
                             output_special_cases),
               "forward correct (special cases)");

    core::Device_Tensor<core::data_type> error_special_cases(
        path_special_cases + "_error");
    core::Host_Tensor<core::data_type> input_gradient_special_cases(
        path_special_cases + "_input_gradient");
    test.check(
        test::compare(softmax_special_cases.backward(error_special_cases),
                      input_gradient_special_cases),
        "backward correct (special cases)");

    // single_value dataset
    std::string path_single_value =
        test.data_path("activations/softmax_single_value");
    core::Device_Tensor<core::data_type> input_single_value(path_single_value +
                                                            "_input");

    activations::Softmax<core::data_type> softmax_single_value(
        device, input_single_value.shape_vec(core::NO_BATCHSIZE));

    core::Host_Tensor<core::data_type> output_single_value(path_single_value +
                                                           "_output");
    test.check(test::compare(softmax_single_value.forward(input_single_value),
                             output_single_value),
               "forward correct (single value)");

    core::Device_Tensor<core::data_type> error_single_value(path_single_value +
                                                            "_error");
    core::Host_Tensor<core::data_type> input_gradient_single_value(
        path_single_value + "_input_gradient");
    test.check(test::compare(softmax_single_value.backward(error_single_value),
                             input_gradient_single_value),
               "backward correct (single value)");
}

/**
 * @brief Launches all tests defined in this file.
 *
 * This function initializes the test framework using command line arguments
 * and runs a series of tests for the activation functions defined in the
 * `activations` namespace.
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
    test.series("Activations");
    test_identity(test);
    test_relu(test);
    test_softmax(test);
    return test.eval();
}
