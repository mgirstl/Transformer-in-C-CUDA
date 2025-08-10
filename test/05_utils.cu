/**
 * @file 05_utils.cu
 * @brief Tests all the classes in the `utils` namespace.
 */
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../src/core.cuh"
#include "../src/test.cuh"
#include "../src/utils.cuh"

/**
 * @brief This function tests the DataLoader class.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_dataloader(test::Test &test) {
    test.component("DataLoader");

    core::Device device;

    auto batchsize = core::Host_Tensor<core::index_type>(
        test.data_path("utils/batchsize"))[0];

    // test without shuffling
    std::string path = test.data_path("utils");
    utils::DataLoader<core::data_type, core::label_type> dataloader(
        device, path + "/data", path + "/target", batchsize, false);

    core::Host_Tensor<core::data_type> data_first_batch(path +
                                                        "/data_first_batch");
    core::Host_Tensor<core::data_type> data_second_batch(path +
                                                         "/data_second_batch");

    core::Host_Tensor<core::label_type> target_first_batch(
        path + "/target_first_batch");
    core::Host_Tensor<core::label_type> target_second_batch(
        path + "/target_second_batch");

    dataloader.next();
    test.check(test::compare(dataloader.data(), data_first_batch),
               "data first batch correct");
    test.check(test::compare(dataloader.target(), target_first_batch),
               "data first batch correct");

    dataloader.next();
    test.check(test::compare(dataloader.data(), data_second_batch),
               "data second batch correct");
    test.check(test::compare(dataloader.target(), target_second_batch),
               "data second batch correct");

    // test with shuffle
    utils::DataLoader<core::data_type, core::label_type> dataloader_shuffle(
        device, path + "/data", path + "/target", batchsize, true);

    dataloader_shuffle.next();
    test.check(!test::compare(dataloader_shuffle.data(), data_first_batch),
               "data first batch shuffle correct");
    test.check(!test::compare(dataloader_shuffle.target(), target_first_batch),
               "data first batch shuffle correct");

    dataloader_shuffle.next();
    test.check(!test::compare(dataloader_shuffle.data(), data_second_batch),
               "data second batch shuffle correct");
    test.check(!test::compare(dataloader_shuffle.target(), target_second_batch),
               "data second batch shuffle correct");

    // Counters
    core::Host_Tensor<core::data_type> complete_data(path + "/data");

    utils::DataLoader<core::data_type, core::index_type> dataloader_counter(
        device, path + "/data", path + "/target", complete_data.batchsize(),
        false, 3);

    // test if max_epoch works as expected
    test.check(dataloader_counter.next(),
               "counter test 1 - correct next return");
    test.check(dataloader_counter.epoch() == 0,
               "counter test 1 - correct epoch");
    test.check(dataloader_counter.iteration() == 0,
               "counter test 1 - correct iteration");
    test.check(dataloader_counter.batch() == 0,
               "counter test 1 - correct batch");

    test.check(dataloader_counter.next(),
               "counter test 1 - correct next return");
    test.check(dataloader_counter.epoch() == 1,
               "counter test 1 - correct epoch");
    test.check(dataloader_counter.iteration() == 1,
               "counter test 1 - correct iteration");
    test.check(dataloader_counter.batch() == 0,
               "counter test 1 - correct batch");

    test.check(dataloader_counter.next(),
               "counter test 1 - correct next return");
    test.check(dataloader_counter.epoch() == 2,
               "counter test 1 - correct epoch");
    test.check(dataloader_counter.iteration() == 2,
               "counter test 1 - correct iteration");
    test.check(dataloader_counter.batch() == 0,
               "counter test 1 - correct batch");

    test.check(!dataloader_counter.next(),
               "counter test 1 - correct next return");
    test.check(dataloader_counter.epoch() == 3,
               "counter test 1 - correct epoch");
    test.check(dataloader_counter.iteration() == 3,
               "counter test 1 - correct iteration");
    test.check(dataloader_counter.batch() == 0,
               "counter test 1 - correct batch");

    // test if max_iteration works as expected
    utils::DataLoader<core::data_type, core::index_type> dataloader_counter_2(
        device, path + "/data", path + "/target", 1, false,
        std::numeric_limits<core::index_type>::max(),
        complete_data.batchsize());
    test.check(dataloader_counter_2.next(),
               "counter test 2 - correct next return");
    test.check(dataloader_counter_2.epoch() == 0,
               "counter test 2 - correct epoch");
    test.check(dataloader_counter_2.iteration() == 0,
               "counter test 2 - correct iteration");
    test.check(dataloader_counter_2.batch() == 0,
               "counter test 2 - correct batch");

    test.check(dataloader_counter_2.next(),
               "counter test 2 - correct next return");
    test.check(dataloader_counter_2.epoch() == 0,
               "counter test 2 - correct epoch");
    test.check(dataloader_counter_2.iteration() == 1,
               "counter test 2 - correct iteration");
    test.check(dataloader_counter_2.batch() == 1,
               "counter test 2 - correct batch");

    while (dataloader_counter_2.next()) {
    }

    test.check(dataloader_counter_2.epoch() == 1,
               "counter test 2 - correct epoch");
    test.check(dataloader_counter_2.iteration() == complete_data.batchsize(),
               "counter test 2 - correct iteration");
    test.check(dataloader_counter_2.batch() == 0,
               "counter test 2 - correct batch");

    test.check(!dataloader_counter_2.next(),
               "counter test 2 - correct next return");
    test.check(dataloader_counter_2.epoch() == 1,
               "counter test 2 - correct epoch");
    test.check(
        dataloader_counter_2.iteration() == complete_data.batchsize() + 1,
        "counter test 2 - correct iteration");
    test.check(dataloader_counter_2.batch() == 1,
               "counter test 2 - correct batch");
}

/**
 * @brief This function tests the Config class.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_config(test::Test &test) {
    test.component("Config");

    int argc = 4;

    std::string path = test.data_path("utils");
    std::string config_path = path + "/config";
    const char *argv[] = {"", "epochs=2", config_path.c_str(), "epochs=3",
                          config_path.c_str()};

    utils::Config<core::data_type> config{argc, argv};

    std::ifstream expected(path + "/config_expected_values");
    if (!expected.is_open())
        throw std::runtime_error(
            "Could not open config_expected_values file: " + path +
            "/config_expected_values");

    std::string line;

    std::getline(expected, line);
    test.check(config.train_data_path == line, "correct train_data_path");

    std::getline(expected, line);
    test.check(config.train_target_path == line, "correct train_target_path");

    std::getline(expected, line);
    test.check(config.test_data_path == line, "correct test_data_path");

    std::getline(expected, line);
    test.check(config.test_target_path == line, "correct test_target_path");

    std::getline(expected, line);
    test.check(config.output_path == line, "correct output_path");

    std::getline(expected, line);
    core::index_type batchsize;
    std::istringstream(line) >> batchsize;
    test.check(config.batchsize == batchsize, "correct batchsize");

    std::getline(expected, line);
    test.check(config.epochs == 3, "correct epochs");

    std::getline(expected, line);
    core::index_type iterations;
    std::istringstream(line) >> iterations;
    test.check(config.iterations == iterations, "correct iterations");

    std::getline(expected, line);
    core::data_type learning_rate;
    std::istringstream(line) >> learning_rate;
    test.check(test::compare(config.learning_rate, learning_rate),
               "correct learning_rate");
}

/**
 * @brief This kernel draws uniform random numbers and saves them in the
 * `tensor` using the provided cuRAND states `rng_state`.
 *
 * @see test_random_number_generator
 */
template <typename curand_state>
__global__ void draw_uniform(core::Kernel_Tensor<core::data_type> tensor,
                             curand_state *rng_state) {
    const core::index_type thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const core::index_type stride = gridDim.x * blockDim.x;

    curand_state _rng_state = rng_state[thread_id];
    for (core::index_type idx = thread_id; idx < tensor.size(); idx += stride)
        tensor[idx] = core::uniform<core::data_type>(_rng_state);

    rng_state[thread_id] = _rng_state;
}

/**
 * @brief This function tests the Random_Number_Generator class by drawing
 * random numbers and comparing there statistical properties with the expected
 * values.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_random_number_generator(test::Test &test) {
    test.component("Random_Number_Generator");

    core::Device device;

    const core::data_type expected_mean = 0.5;
    const core::data_type expected_variance = 1. / 12;
    core::Device_Tensor<core::data_type> tensor{{1'000'000}};

    // test philox
    auto &philox = utils::Random_Number_Generator<
        curandStatePhilox4_32_10_t>::get_instance(device);

    draw_uniform<<<device.blocks(), device.threads()>>>(tensor,
                                                        philox.states());
    core::checkCuda(cudaGetLastError());
    core::checkCuda(cudaDeviceSynchronize());

    test.check(test::compare_mean(tensor, expected_mean, expected_variance),
               "Philox correct mean");
    test.check(test::compare_variance(tensor, expected_mean, expected_variance),
               "Philox correct variance");
    test.comment("Philox Random Numbers: " + tensor.str());

    // test xorwow
    auto &xorwow =
        utils::Random_Number_Generator<curandStateXORWOW_t>::get_instance(
            device);

    draw_uniform<<<device.blocks(), device.threads()>>>(tensor,
                                                        xorwow.states());
    core::checkCuda(cudaGetLastError());
    core::checkCuda(cudaDeviceSynchronize());

    test.check(
        test::compare_mean<core::data_type>(tensor, expected_mean,
                                            200. / 9. * expected_variance),
        "Xorwow correct mean");  // need a larger confidence interval here
    test.check(test::compare_variance(tensor, expected_mean, expected_variance),
               "Xorwow correct variance");
    test.comment("Xorwow Random Numbers: " + tensor.str());
}

/**
 * @brief This function tests the Argmax class using random data.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_argmax(test::Test &test) {
    test.component("Argmax");

    core::Device device;

    std::string path = test.data_path("utils/argmax");
    core::Device_Tensor<core::data_type> input(path + "_input");
    core::Host_Tensor<core::index_type> output(path + "_output");

    utils::Argmax<core::data_type, core::index_type> argmax(
        device, input.shape_vec(core::NO_BATCHSIZE));

    test.check(test::compare(argmax.apply(input), output), "apply correct");
}

/**
 * @brief This function tests the Indicator class using random data.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_indicator(test::Test &test) {
    test.component("Indicator");

    core::Device device;

    std::string path = test.data_path("utils/indicator");
    core::Device_Tensor<core::index_type> input(path + "_input");
    core::Device_Tensor<core::index_type> target(path + "_target");
    core::Host_Tensor<core::mask_type> output(path + "_output");

    utils::Indicator<core::index_type> indicator(
        device, input.shape_vec(core::NO_BATCHSIZE));

    test.check(test::compare(indicator.apply(input, target), output),
               "apply correct");
}

/**
 * @brief Launches all tests defined in this file.
 *
 * This function initializes the test framework using command line arguments
 * and runs a series of tests for the classes defined in the `utils` namespace.
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
    test.series("Utils");
    test_dataloader(test);
    test_config(test);
    test_random_number_generator(test);
    test_indicator(test);
    test_argmax(test);
    return test.eval();
}
