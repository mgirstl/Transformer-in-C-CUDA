/**
 * @file 04_optimizers.cu
 * @brief Tests all the classes in the `optimizers` namespace.
 */

#include <string>
#include <unordered_map>

#include "../src/core.cuh"
#include "../src/optimizers.cuh"
#include "../src/test.cuh"

/**
 * @brief This function tests the None optimizer using random data.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_none(test::Test &test) {
    test.component("None");

    core::Device device;

    std::string path = test.data_path("optimizers/none");
    core::Device_Tensor<core::data_type> weights(path + "_weights");

    std::unordered_map<std::string, core::data_type> kwargs{
        {"learning_rate", 1}};

    optimizers::None<core::data_type> none(device, weights.shape_vec(), kwargs);

    core::Device_Tensor<core::data_type> gradient(path + "_gradient");
    core::Host_Tensor<core::data_type> updated_weights(path +
                                                       "_updated_weights");
    none.update(weights, gradient);
    test.check(test::compare(weights, updated_weights), "update correct");
}

/**
 * @brief This function tests the SGD optimizer using random data.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_sgd(test::Test &test) {
    test.component("SGD");

    core::Device device;

    std::string path = test.data_path("optimizers/sgd");
    core::Device_Tensor<core::data_type> weights(path + "_weights");

    auto learning_rate =
        core::Host_Tensor<core::data_type>(path + "_learning_rate")[0];

    std::unordered_map<std::string, core::data_type> kwargs{
        {"learning_rate", learning_rate}};

    optimizers::SGD<core::data_type> sgd(device, weights.shape_vec(), kwargs);

    core::Device_Tensor<core::data_type> gradient(path + "_gradient");
    core::Host_Tensor<core::data_type> updated_weights(path +
                                                       "_updated_weights");
    sgd.update(weights, gradient);
    test.check(test::compare(weights, updated_weights), "update correct");
}

/**
 * @brief This function tests the Adam optimizer using random data.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_adam(test::Test &test) {
    test.component("Adam");

    core::Device device;

    std::string path = test.data_path("optimizers/adam");
    core::Device_Tensor<core::data_type> weights(path + "_weights");

    auto learning_rate =
        core::Host_Tensor<core::data_type>(path + "_learning_rate")[0];
    auto mu = core::Host_Tensor<core::data_type>(path + "_mu")[0];
    auto rho = core::Host_Tensor<core::data_type>(path + "_rho")[0];

    std::unordered_map<std::string, core::data_type> kwargs{
        {"learning_rate", learning_rate}, {"mu", mu}, {"rho", rho}};

    optimizers::Adam<core::data_type> adam(device, weights.shape_vec(), kwargs);

    // first step
    core::Device_Tensor<core::data_type> gradient(path + "_gradient_step_1");
    adam.update(weights, gradient);
    core::Host_Tensor<core::data_type> updated_weights_step_1(
        path + "_updated_weights_step_1");
    test.check(test::compare(weights, updated_weights_step_1),
               "weights update correct - first step");
    test.comment("Computed: " + weights.str());
    test.comment("Expected: " + updated_weights_step_1.str());

    core::Host_Tensor<core::data_type> first_moment_step_1(
        path + "_first_momentum_step_1");
    core::Host_Tensor<core::data_type> second_moment_step_1(
        path + "_second_momentum_step_1");

    test.check(test::compare(adam.get_first_moment(), first_moment_step_1),
               "first moment correct - first step");
    test.check(test::compare(adam.get_second_moment(), second_moment_step_1),
               "second moment correct - first step");

    // second step
    gradient.load(path + "_gradient_step_2");
    adam.update(weights, gradient);
    core::Host_Tensor<core::data_type> updated_weights_step_2(
        path + "_updated_weights_step_2");
    test.check(test::compare(weights, updated_weights_step_2),
               "weights update correct - second step");
    test.comment("Computed: " + weights.str());
    test.comment("Expected: " + updated_weights_step_2.str());

    core::Host_Tensor<core::data_type> first_moment_step_2(
        path + "_first_momentum_step_2");
    core::Host_Tensor<core::data_type> second_moment_step_2(
        path + "_second_momentum_step_2");

    test.check(test::compare(adam.get_first_moment(), first_moment_step_2),
               "first moment correct - second step");
    test.check(test::compare(adam.get_second_moment(), second_moment_step_2),
               "second moment correct - second step");
}

/**
 * @brief This function tests the Noam scheduler using random data and by
 * checking the changing learning rate.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_noam(test::Test &test) {
    test.component("Noam");

    core::Device device;

    std::string path = test.data_path("optimizers/noam");
    auto model_dim = core::Host_Tensor<core::data_type>(path + "_model_dim")[0];
    auto warmup_steps =
        core::Host_Tensor<core::data_type>(path + "_warmup_steps")[0];

    std::unordered_map<std::string, core::data_type> kwargs{
        {"model_dim", model_dim}, {"warmup_steps", warmup_steps}};

    // normal update
    core::Device_Tensor<core::data_type> weights(path + "_weights");

    optimizers::Noam<core::data_type, optimizers::SGD> noam(
        device, weights.shape_vec(), kwargs);

    core::Device_Tensor<core::data_type> gradient(path + "_gradient");
    noam.update(weights, gradient);
    core::Host_Tensor<core::data_type> updated_weights(path +
                                                       "_updated_weights");
    test.check(test::compare(weights, updated_weights), "update correct");

    // learning rate calculation
    optimizers::Noam<core::data_type, optimizers::None> noam2(device, {},
                                                              kwargs);
    core::Device_Tensor<core::data_type> tensor;

    core::index_type current_step = 0;
    auto num_tests = core::Host_Tensor<int>(path + "_num_tests")[0];
    for (int i = 1; i <= num_tests; ++i) {
        auto step = core::Host_Tensor<core::data_type>(path + "_step_" +
                                                       std::to_string(i))[0];
        auto expected_learning_rate = core::Host_Tensor<core::data_type>(
            path + "_learning_rate_" + std::to_string(i))[0];

        for (; current_step < step; ++current_step)
            noam2.update(tensor, tensor);

        auto learning_rate = noam2.get_learning_rate();

        test.check(test::compare(learning_rate, expected_learning_rate),
                   "learning rate correct - step " + std::to_string(i));
    }
}

/**
 * @class Test_Class
 * @brief A dummy class to test if the `optimizer_type` concept is implemented
 * correctly.
 *
 * @see test_optimizer_template
 */
template <core::arithmetic_type data_type,
          template <typename> class optimizer_type>
class Test_Class {
    static_assert(
        optimizers::optimizer_type<optimizer_type<data_type>, data_type>,
        "optimizer_type must satisfy the optimizers::optimizer_type concept!");
};


/**
 * @brief This function tests if the different optimizers fulfill the
 * `optimizer_type` concept.
 *
 * @note This test automatically succeeds if the compilation is successful.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_optimizer_template(test::Test &test) {
    test.component("Optimizer Concept");

    #pragma diag_suppress declared_but_not_referenced

        volatile Test_Class<core::data_type, optimizers::None> a;
        volatile Test_Class<core::data_type, optimizers::SGD> b;
        volatile Test_Class<core::data_type, optimizers::Adam> c;
        volatile Test_Class<core::data_type,
            optimizers::None_with_Noam_Scheduler> d;
        volatile Test_Class<core::data_type,
            optimizers::SGD_with_Noam_Scheduler> e;
        volatile Test_Class<core::data_type,
            optimizers::Adam_with_Noam_Scheduler> f;

        // The following combinations do not work (non-comprehensive list):
        // volatile Test_Class<core::data_type,
        //     optimizers::Adam<core::data_type>> g;
        // volatile Test_Class<core::data_type, optimizers::Noam> h;
        // volatile Test_Class<core::data_type,
        //     optimizers::Noam<optimizers::Adam>> i;
        // volatile Test_Class<core::data_type,
        //     optimizers::Noam<core::data_type,optimizers::Adam>> j;
        // volatile Test_Class<core::data_type, optimizers::Noam<
        //     core::data_type, optimizers::Adam<core::data_type>>> k;

    #pragma diag_default declared_but_not_referenced

    test.check(true, "Compiled all combinations!");
}

/**
 * @brief Launches all tests defined in this file.
 *
 * This function initializes the test framework using command line arguments
 * and runs a series of tests for the optimizers defined in the `optimizers`
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
    test.series("Optimizers");
    test_none(test);
    test_sgd(test);
    test_adam(test);
    test_noam(test);
    test_optimizer_template(test);
    return test.eval();
}
