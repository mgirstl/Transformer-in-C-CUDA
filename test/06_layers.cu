/**
 * @file layers.cu
 * @brief Tests all the classes in the `layers` namespace.
 */

#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "../src/core.cuh"
#include "../src/layers.cuh"
#include "../src/optimizers.cuh"
#include "../src/test.cuh"

constexpr core::data_type epsilon =
    std::numeric_limits<core::data_type>::epsilon() *
    std::numeric_limits<core::data_type>::epsilon();

/**
 * @brief This function tests the initialization of the Dense layer by checking
 * its statistical properties.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_dense_initialization(test::Test &test) {
    test.component("Dense - Initialization");

    core::Device device;

    std::vector<core::index_type> input_shape{10, 20, 40};
    std::vector<core::index_type> output_shape{10, 20, 30};

    layers::Dense<core::data_type, optimizers::None> dense_1(
        device, input_shape, output_shape);

    auto weights_shape = dense_1.get_weights().shape_vec();
    test.check(weights_shape[0] == 40 && weights_shape[1] == 30,
               "weights correct shape");
    test.check(
        dense_1.get_bias().rank() == 1 && dense_1.get_bias().size() == 30,
        "bias correct shape");

    auto weights = core::Host_Tensor(dense_1.get_weights());
    test.check(!test::compare(weights[0], weights[1]),
               "bias components are different");

    auto bias = core::Host_Tensor(dense_1.get_bias());
    test.check(!test::compare(bias[0], bias[1]),
               "bias components are different");

    test.check(test::compare_mean<core::data_type>(weights, 0, 2. / 40.),
               "weights correct mean");
    test.check(test::compare_variance<core::data_type>(weights, 0, 2. / 40.),
               "weights correct variance");
    test.check(test::compare_mean<core::data_type>(bias, 0, 2. / 40.),
               "bias correct mean");
    test.check(test::compare_variance<core::data_type>(bias, 0, 2. / 40.),
               "bias correct variance");

    layers::Dense<core::data_type, optimizers::None> dense_2(
        device, input_shape, output_shape);

    test.check(!test::compare(dense_1.get_weights(), dense_2.get_weights()),
               "weights are different for different instances");

    test.check(!test::compare(dense_1.get_bias(), dense_2.get_bias()),
               "bias is different for different instances");

    test.comment("weights first instance: " + dense_1.get_weights().str());
    test.comment("weights second instance: " + dense_2.get_weights().str());
    test.comment("bias first instance: " + dense_1.get_bias().str());
    test.comment("bias second instance: " + dense_2.get_bias().str());

    // save / load test
    dense_2.save(test.temporary_path("dense"));

    layers::Dense<core::data_type, optimizers::None> dense_3(
        device, input_shape, output_shape);

    dense_3.load(test.temporary_path("dense"), 0);
    core::checkCuda(cudaDeviceSynchronize());

    test.check(test::compare(dense_2.get_weights(), dense_3.get_weights()),
               "weights  load correct");
    test.check(test::compare(dense_2.get_bias(), dense_3.get_bias()),
               "bias load correct");
}

/**
 * @brief This function tests the Dense layer without bias using random data.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_dense_without_bias(test::Test &test) {
    test.component("Dense - without bias");
    std::string path = test.data_path("layers/dense_without_bias");

    core::Device device;

    auto learning_rate =
        core::Host_Tensor<core::data_type>(path + "_learning_rate")[0];
    std::unordered_map<std::string, core::data_type> optimizer_kw{
        {"learning_rate", learning_rate}};

    core::Device_Tensor<core::data_type> input(path + "_input");
    core::Host_Tensor<core::data_type> output(path + "_output");
    core::Device_Tensor<core::data_type> error(path + "_error");
    core::Host_Tensor<core::data_type> input_gradient(path + "_input_gradient");

    layers::Dense<core::data_type, optimizers::SGD> dense(
        device, input.shape_vec(core::NO_BATCHSIZE),
        output.shape_vec(core::NO_BATCHSIZE), optimizer_kw, false);

    test.check(
        test::compare(dense.get_bias(), core::Device_Tensor<core::data_type>()),
        "bias correct before loading");

    dense.load(path);

    test.check(
        test::compare(dense.get_bias(), core::Device_Tensor<core::data_type>()),
        "bias correct after loading");

    test.check(test::compare(dense.forward(input), output), "forward correct");
    test.check(test::compare(dense.backward(error), input_gradient),
               "backward error propagation correct");

    auto weights = dense.get_weights();
    dense.load(path + "_updated");

    test.check(test::compare(weights, dense.get_weights()),
               "weights update correct");
}

/**
 * @brief This function tests the Dense layer with bias using random data.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_dense_with_bias(test::Test &test) {
    test.component("Dense - without bias");
    std::string path = test.data_path("layers/dense_with_bias");

    core::Device device;

    auto learning_rate =
        core::Host_Tensor<core::data_type>(path + "_learning_rate")[0];
    std::unordered_map<std::string, core::data_type> optimizer_kw{
        {"learning_rate", learning_rate}};

    core::Device_Tensor<core::data_type> input(path + "_input");
    core::Host_Tensor<core::data_type> output(path + "_output");
    core::Device_Tensor<core::data_type> error(path + "_error");
    core::Host_Tensor<core::data_type> input_gradient(path + "_input_gradient");

    layers::Dense<core::data_type, optimizers::SGD> dense(
        device, input.shape_vec(core::NO_BATCHSIZE),
        output.shape_vec(core::NO_BATCHSIZE), optimizer_kw, true);

    dense.load(path);

    test.check(test::compare(dense.forward(input), output), "forward correct");
    test.check(test::compare(dense.backward(error), input_gradient),
               "backward error propagation correct");

    auto weights = dense.get_weights();
    auto bias = dense.get_bias();
    dense.load(path + "_updated");
    test.check(test::compare(weights, dense.get_weights()),
               "weights update correct");
    test.check(test::compare(bias, dense.get_bias()), "bias update correct");
}

/**
 * @brief This function tests the Reshape layer using random data.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_reshape(test::Test &test) {
    test.component("Reshape");
    std::string path = test.data_path("layers/reshape");

    core::Device_Tensor<core::data_type> input(path + "_input");
    core::Host_Tensor<core::data_type> output(path + "_output");
    core::Device_Tensor<core::data_type> error(path + "_error");
    core::Host_Tensor<core::data_type> input_gradient(path + "_input_gradient");

    layers::Reshape<core::data_type> reshape(
        input.shape_vec(core::NO_BATCHSIZE),
        output.shape_vec(core::NO_BATCHSIZE));

    test.check(test::compare(reshape.forward(input), output),
               "forward correct");
    test.check(test::compare(reshape.backward(error), input_gradient),
               "backward correct");

    // test if runtime error works
    try {
        layers::Reshape<core::data_type> reshape_2({1, 2}, {2, 2});
    } catch (const std::invalid_argument &e) {
        test.check(true, "reshape runtime error");
    }
}

/**
 * @brief This function tests the Upscale layer using random data.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_upscale(test::Test &test) {
    test.component("Upscale");

    std::string path = test.data_path("layers/upscale");

    core::Device device;

    core::Device_Tensor<core::data_type> input;
    core::Host_Tensor<core::data_type> output;

    // identity
    input.load(path + "_identity_input");
    output.load(path + "_identity_output");

    layers::Upscale<core::data_type> identity(
        device, output.shape_vec(core::NO_BATCHSIZE));

    test.check(test::compare(identity.forward(input), output),
               "forward identity correct");

    // small
    input.load(path + "_small_input");
    output.load(path + "_small_input");

    layers::Upscale<core::data_type> small(
        device, output.shape_vec(core::NO_BATCHSIZE));

    test.check(test::compare(small.forward(input), output),
               "forward small correct");

    // square
    input.load(path + "_square_input");
    output.load(path + "_square_input");

    layers::Upscale<core::data_type> square(
        device, output.shape_vec(core::NO_BATCHSIZE));

    test.check(test::compare(square.forward(input), output),
               "forward square correct");

    // rectangular
    input.load(path + "_rectangular_input");
    output.load(path + "_rectangular_input");

    layers::Upscale<core::data_type> rectangular(
        device, output.shape_vec(core::NO_BATCHSIZE));

    test.check(test::compare(rectangular.forward(input), output),
               "forward rectangular correct");
}

/**
 * @brief This function tests the Dropout layer using random data and
 * statistical tests.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_dropout(test::Test &test) {
    test.component("Dropout");

    std::string path = test.data_path("layers/dropout");

    core::Device device;

    core::data_type probability =
        core::Host_Tensor<core::data_type>(path + "_probability")[0];
    core::data_type probability_one = 1 - probability;

    core::Device_Tensor<core::data_type> input(path + "_input");
    core::Device_Tensor<core::data_type> error(path + "_error");

    auto train_scaling =
        core::Host_Tensor<core::data_type>(path + "_train_scaling")[0];
    core::Host_Tensor<core::data_type> train_input_gradient(
        path + "_train_input_gradient");
    core::Host_Tensor<core::data_type> eval_input_gradient(
        path + "_eval_input_gradient");
    core::Host_Tensor<core::data_type> eval_output(path + "_eval_output");

    layers::Dropout<core::data_type> dropout(
        device, input.shape_vec(core::NO_BATCHSIZE), probability);

    auto output = dropout.forward(input);
    auto mask = dropout.get_mask();

    // statistical properties
    core::data_type var =
        probability * probability_one;  // bernoulli distribution
    core::data_type mean = probability_one;
    test.check(test::compare_mean(mask, mean, var), "probability correct");
    test.check(test::compare_variance(mask, mean, var), "variance correct");

    // scaling
    core::Host_Tensor<core::data_type> _input{input};
    core::Host_Tensor<core::data_type> _output{output};
    core::Host_Tensor<core::data_type> _mask{mask};
    core::data_type sum = 0;
    core::index_type total = 0;
    for (core::index_type idx = 0; idx < _input.size(); ++idx) {
        if (_mask[idx] > 0.5) {
            ++total;
            sum += _input[idx] / _output[idx];
        }
    }
    test.check(test::compare<core::data_type>(sum / total, train_scaling, 1000),
               "training correct scaling");

    // unique mask - second step
    dropout.forward(input);
    test.check(!test::compare(dropout.get_mask(), mask),
               "second step different mask");

    // unique mask - second instance
    layers::Dropout<core::data_type> dropout_2(
        device, input.shape_vec(core::NO_BATCHSIZE), probability);

    dropout_2.forward(input);
    test.check(!test::compare(dropout_2.get_mask(), mask),
               "second instance different mask");
    test.check(!test::compare(dropout_2.get_mask(), dropout.get_mask()),
               "second instance different mask");

    // backward pass
    mask.load(path + "_mask");
    dropout.set_mask(mask);
    test.check(test::compare(dropout.backward(error), train_input_gradient),
               "backward correct");

    // evaluation mode
    dropout.eval();
    test.check(test::compare(dropout.forward(input), eval_output),
               "forward evaluation correct");
    test.check(test::compare(dropout.backward(error), eval_input_gradient),
               "backward evaluation correct");
}

/**
 * @brief This function tests the LayerNorm layer using random data.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_layernorm(test::Test &test) {
    test.component("LayerNorm");
    std::string path = test.data_path("layers/layernorm");

    core::Device device;

    auto learning_rate =
        core::Host_Tensor<core::data_type>(path + "_learning_rate")[0];
    std::unordered_map<std::string, core::data_type> optimizer_kw{
        {"learning_rate", learning_rate}};

    core::Host_Tensor<core::index_type> _normalized_shape(path +
                                                          "_normalized_shape");
    std::vector<core::index_type> normalized_shape(_normalized_shape.size());
    for (core::index_type idx = 0; idx < _normalized_shape.size(); ++idx)
        normalized_shape[idx] = _normalized_shape[idx];

    core::Device_Tensor<core::data_type> input(path + "_input");
    core::Host_Tensor<core::data_type> output(path + "_output");
    core::Host_Tensor<core::data_type> input_normalized(path +
                                                        "_input_normalized");
    core::Device_Tensor<core::data_type> error(path + "_error");
    core::Host_Tensor<core::data_type> input_gradient(path + "_input_gradient");

    layers::LayerNorm<core::data_type, optimizers::SGD> layer(
        device, input.shape_vec(core::NO_BATCHSIZE), normalized_shape,
        optimizer_kw);

    // initialization
    auto gamma = layer.get_gamma();
    auto beta = layer.get_beta();
    layer.load(path);

    test.check(test::compare(gamma, layer.get_gamma()),
               "gamma initialization correct");
    test.check(test::compare(beta, layer.get_beta()),
               "beta initialization correct");

    // forward pass
    test.check(test::compare(layer.forward(input), output), "forward correct");
    test.check(test::compare(layer.get_input_normalized(), input_normalized),
               "input normalized correct");

    // backward pass
    test.check(test::compare(layer.backward(error), input_gradient),
               "backward error propagation correct");

    gamma = layer.get_gamma();
    beta = layer.get_beta();
    layer.load(path + "_updated");

    test.check(test::compare(gamma, layer.get_gamma()), "gamma update correct");
    test.check(test::compare(beta, layer.get_beta()), "beta update correct");
}

/**
 * @brief This function tests the Embedding layer using random data.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_embedding(test::Test &test) {
    test.component("Embedding");
    std::string path = test.data_path("layers/embedding");

    core::Device device;

    auto learning_rate =
        core::Host_Tensor<core::data_type>(path + "_learning_rate")[0];
    std::unordered_map<std::string, core::data_type> optimizer_kw{
        {"learning_rate", learning_rate}};

    auto num_embeddings =
        core::Host_Tensor<core::label_type>(path + "_num_embeddings")[0];

    core::Device_Tensor<core::label_type> input(path + "_input");
    core::Host_Tensor<core::data_type> output(path + "_output");
    core::Device_Tensor<core::data_type> error(path + "_error");

    layers::Embedding<core::label_type, core::data_type, optimizers::SGD> layer(
        device, input.shape_vec(core::NO_BATCHSIZE),
        output.shape_vec(core::NO_BATCHSIZE), num_embeddings, optimizer_kw);

    // statistics
    test.check(test::compare_mean<core::data_type>(layer.get_weights(), 0,
                                                   2. / num_embeddings),
               "weights correct mean");
    test.check(test::compare_variance<core::data_type>(layer.get_weights(), 0,
                                                       2. / num_embeddings),
               "weights correct variance");

    // preparation
    layer.load(path);

    // forward pass
    test.check(test::compare(layer.forward(input), output), "forward correct");

    // backward pass
    layer.backward(error);

    auto weights = layer.get_weights();
    layer.load(path + "_updated");

    test.check(test::compare(weights, layer.get_weights()),
               "weights update correct");
}

/**
 * @brief This function tests the PositionwiseFeedForward layer using random
 * data.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_positionwisefeedforward(test::Test &test) {
    test.component("PositionwiseFeedForward");
    std::string path = test.data_path("layers/positionwisefeedforward");

    core::Device device;

    auto learning_rate =
        core::Host_Tensor<core::data_type>(path + "_learning_rate")[0];
    std::unordered_map<std::string, core::data_type> optimizer_kw{
        {"learning_rate", learning_rate}};

    auto dropout = core::Host_Tensor<core::data_type>(path + "_dropout")[0];
    if (dropout < epsilon) test.comment("We test here without dropout!");

    auto hidden_dim =
        core::Host_Tensor<core::index_type>(path + "_hidden_dim")[0];

    core::Device_Tensor<core::data_type> input(path + "_input");
    core::Host_Tensor<core::data_type> output(path + "_output");
    core::Device_Tensor<core::data_type> error(path + "_error");
    core::Host_Tensor<core::data_type> input_gradient(path + "_input_gradient");

    auto hidden_shape = input.shape_vec(core::NO_BATCHSIZE);
    hidden_shape.back() = hidden_dim;

    layers::PositionwiseFeedForward<core::data_type, optimizers::SGD> layer(
        device, input.shape_vec(core::NO_BATCHSIZE), hidden_shape, optimizer_kw,
        dropout);

    layer.load(path);

    // forward and backward pass
    test.check(test::compare(layer.forward(input), output), "forward correct");
    test.check(test::compare(layer.backward(error), input_gradient),
               "backward error propagation correct");

    // check update
    const auto freeze = layer;
    layer.load(path + "_updated");

    test.check(test::compare(freeze.get_dense_1().get_weights(),
                             layer.get_dense_1().get_weights()),
               "dense_1 weights update correct");
    test.check(test::compare(freeze.get_dense_1().get_bias(),
                             layer.get_dense_1().get_bias()),
               "dense_1 bias update correct");
    test.check(test::compare(freeze.get_dense_2().get_weights(),
                             layer.get_dense_2().get_weights()),
               "dense_2 weights update correct");
    test.check(test::compare(freeze.get_dense_2().get_bias(),
                             layer.get_dense_2().get_bias()),
               "dense_2 bias update correct");
}

/**
 * @brief This function tests the PositionalEncoding layer using random data.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_positionalencoding(test::Test &test) {
    test.component("PositionalEncoding");
    std::string path = test.data_path("layers/positionalencoding");

    core::Device device;

    auto dropout = core::Host_Tensor<core::data_type>(path + "_dropout")[0];
    if (dropout < epsilon) test.comment("We test here without dropout!");

    auto max_batchsize =
        core::Host_Tensor<core::index_type>(path + "_max_batchsize")[0];

    core::Device_Tensor<core::data_type> input(path + "_input");
    core::Host_Tensor<core::data_type> pe(path + "_pe");
    core::Host_Tensor<core::data_type> output(path + "_output");
    core::Device_Tensor<core::data_type> error(path + "_error");
    core::Host_Tensor<core::data_type> input_gradient(path + "_input_gradient");

    layers::PositionalEncoding<core::data_type> layer(
        device, input.shape_vec(core::NO_BATCHSIZE), dropout, max_batchsize);

    // initialization
    test.check(test::compare(layer.get_pe(), pe), "pe correct");

    // forward and backward pass
    test.check(test::compare(layer.forward(input), output), "forward correct");
    test.check(test::compare(layer.backward(error), input_gradient),
               "backward error propagation correct");
}

/**
 * @brief This function tests the MultiheadAttention layer using random data.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_multiheadattention(test::Test &test) {
    test.component("MultiheadAttention");
    std::string path = test.data_path("layers/multiheadattention");

    core::Device device;

    auto learning_rate =
        core::Host_Tensor<core::data_type>(path + "_learning_rate")[0];
    std::unordered_map<std::string, core::data_type> optimizer_kw{
        {"learning_rate", learning_rate}};

    auto dropout = core::Host_Tensor<core::data_type>(path + "_dropout")[0];
    if (dropout < epsilon) test.comment("We test here without dropout!");

    auto num_heads =
        core::Host_Tensor<core::index_type>(path + "_num_heads")[0];

    core::Device_Tensor<core::data_type> query(path + "_query");
    core::Device_Tensor<core::data_type> key(path + "_key");
    core::Device_Tensor<core::data_type> value(path + "_value");
    core::Device_Tensor<core::mask_type> mask(path + "_mask");
    core::Host_Tensor<core::data_type> output(path + "_output");
    core::Device_Tensor<core::data_type> error(path + "_error");
    core::Host_Tensor<core::data_type> query_gradient(path + "_query_gradient");
    core::Host_Tensor<core::data_type> key_gradient(path + "_key_gradient");
    core::Host_Tensor<core::data_type> value_gradient(path + "_value_gradient");

    layers::MultiheadAttention<core::data_type, optimizers::SGD> layer(
        device, query.shape_vec(core::NO_BATCHSIZE),
        key.shape_vec(core::NO_BATCHSIZE), optimizer_kw, num_heads, dropout);

    layer.load(path);

    // forward and backward pass
    test.check(test::compare(layer.forward(query, key, value, mask), output),
               "forward correct");
    layer.backward(error);
    test.check(test::compare(layer.get_query_gradient(), query_gradient),
               "query gradient correct");
    test.check(test::compare(layer.get_key_gradient(), key_gradient),
               "key gradient correct");
    test.check(test::compare(layer.get_value_gradient(), value_gradient),
               "value gradient correct");

    // check update
    const auto freeze = layer;
    layer.load(path + "_updated");

    test.check(test::compare(freeze.get_query_linear().get_weights(),
                             layer.get_query_linear().get_weights()),
               "query_linear weights update correct");
    test.check(test::compare(freeze.get_key_linear().get_weights(),
                             layer.get_key_linear().get_weights()),
               "key_linear bias update correct");
    test.check(test::compare(freeze.get_value_linear().get_weights(),
                             layer.get_value_linear().get_weights()),
               "value_linear weights update correct");
    test.check(test::compare(freeze.get_output_linear().get_weights(),
                             layer.get_output_linear().get_weights()),
               "output_linear weights update correct");
    test.check(test::compare(freeze.get_output_linear().get_bias(),
                             layer.get_output_linear().get_bias()),
               "output_linear bias update correct");
}

/**
 * @brief This function tests the EncoderLayer layer using random data.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_encoderlayer(test::Test &test) {
    test.component("EncoderLayer");
    std::string path = test.data_path("layers/encoderlayer");

    core::Device device;

    auto learning_rate =
        core::Host_Tensor<core::data_type>(path + "_learning_rate")[0];
    std::unordered_map<std::string, core::data_type> optimizer_kw{
        {"learning_rate", learning_rate}};

    auto dropout = core::Host_Tensor<core::data_type>(path + "_dropout")[0];
    if (dropout < epsilon) test.comment("We test here without dropout!");

    auto num_heads =
        core::Host_Tensor<core::index_type>(path + "_num_heads")[0];

    auto hidden_dim =
        core::Host_Tensor<core::index_type>(path + "_hidden_dim")[0];

    core::Device_Tensor<core::data_type> input(path + "_input");
    core::Device_Tensor<core::mask_type> mask(path + "_mask");
    core::Host_Tensor<core::data_type> output(path + "_output");
    core::Device_Tensor<core::data_type> error(path + "_error");
    core::Host_Tensor<core::data_type> input_gradient(path + "_input_gradient");

    auto hidden_shape = input.shape_vec(core::NO_BATCHSIZE);
    hidden_shape.back() = hidden_dim;

    layers::EncoderLayer<core::data_type, optimizers::SGD> layer(
        device, input.shape_vec(core::NO_BATCHSIZE), hidden_shape, optimizer_kw,
        num_heads, dropout);

    layer.load(path);

    // forward and backward pass
    test.check(test::compare(layer.forward(input, mask), output),
               "forward correct");
    test.check(test::compare(layer.backward(error), input_gradient),
               "backward error propagation correct");

    // check update
    const auto freeze = layer;
    layer.load(path + "_updated");

    test.check(test::compare(freeze.get_mha().get_query_linear().get_weights(),
                             layer.get_mha().get_query_linear().get_weights()),
               "mha query_linear weights update correct");
    test.check(test::compare(freeze.get_mha().get_key_linear().get_weights(),
                             layer.get_mha().get_key_linear().get_weights()),
               "mha key_linear bias update correct");
    test.check(test::compare(freeze.get_mha().get_value_linear().get_weights(),
                             layer.get_mha().get_value_linear().get_weights()),
               "mha value_linear weights update correct");
    test.check(test::compare(freeze.get_mha().get_output_linear().get_weights(),
                             layer.get_mha().get_output_linear().get_weights()),
               "mha output_linear weights update correct");
    test.check(test::compare(freeze.get_mha().get_output_linear().get_bias(),
                             layer.get_mha().get_output_linear().get_bias()),
               "mha output_linear bias update correct");
    test.check(test::compare(freeze.get_mha_norm().get_gamma(),
                             layer.get_mha_norm().get_gamma()),
               "mha norm gamma update correct");
    test.check(test::compare(freeze.get_mha_norm().get_beta(),
                             layer.get_mha_norm().get_beta()),
               "mha norm beta update correct");
    test.check(test::compare(freeze.get_pwff().get_dense_1().get_weights(),
                             layer.get_pwff().get_dense_1().get_weights()),
               "pwff dense_1 weights update correct");
    test.check(test::compare(freeze.get_pwff().get_dense_1().get_bias(),
                             layer.get_pwff().get_dense_1().get_bias()),
               "pwff dense_1 bias update correct");
    test.check(test::compare(freeze.get_pwff().get_dense_2().get_weights(),
                             layer.get_pwff().get_dense_2().get_weights()),
               "pwff dense_2 weights update correct");
    test.check(test::compare(freeze.get_pwff().get_dense_2().get_bias(),
                             layer.get_pwff().get_dense_2().get_bias()),
               "pwff dense_2 bias update correct");
    test.check(test::compare(freeze.get_pwff_norm().get_gamma(),
                             layer.get_pwff_norm().get_gamma()),
               "pwff norm gamma update correct");
    test.check(test::compare(freeze.get_pwff_norm().get_beta(),
                             layer.get_pwff_norm().get_beta()),
               "pwff norm beta update correct");
}

/**
 * @brief This function tests the DecoderLayer layer using random data.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_decoderlayer(test::Test &test) {
    test.component("DecoderLayer");
    std::string path = test.data_path("layers/decoderlayer");

    core::Device device;

    auto learning_rate =
        core::Host_Tensor<core::data_type>(path + "_learning_rate")[0];
    std::unordered_map<std::string, core::data_type> optimizer_kw{
        {"learning_rate", learning_rate}};

    auto dropout = core::Host_Tensor<core::data_type>(path + "_dropout")[0];
    if (dropout < epsilon) test.comment("We test here without dropout!");

    auto num_heads =
        core::Host_Tensor<core::index_type>(path + "_num_heads")[0];

    auto hidden_dim =
        core::Host_Tensor<core::index_type>(path + "_hidden_dim")[0];

    core::Device_Tensor<core::data_type> target(path + "_target");
    core::Device_Tensor<core::mask_type> target_mask(path + "_target_mask");
    core::Device_Tensor<core::data_type> source(path + "_source");
    core::Device_Tensor<core::mask_type> source_mask(path + "_source_mask");
    core::Host_Tensor<core::data_type> output(path + "_output");
    core::Host_Tensor<core::data_type> attention(path + "_attention");
    core::Device_Tensor<core::data_type> error(path + "_error");
    core::Host_Tensor<core::data_type> target_gradient(path +
                                                       "_target_gradient");
    core::Host_Tensor<core::data_type> source_gradient(path +
                                                       "_source_gradient");

    auto hidden_shape = target.shape_vec(core::NO_BATCHSIZE);
    hidden_shape.back() = hidden_dim;

    layers::DecoderLayer<core::data_type, optimizers::SGD> layer(
        device, target.shape_vec(core::NO_BATCHSIZE),
        source.shape_vec(core::NO_BATCHSIZE), hidden_shape, optimizer_kw,
        num_heads, dropout);

    layer.load(path);

    // forward and backward pass
    test.check(
        test::compare(layer.forward(target, target_mask, source, source_mask),
                      output),
        "forward correct");
    test.check(test::compare(layer.get_attention(), attention),
               "attention correct");
    layer.backward(error);
    test.check(test::compare(layer.get_target_gradient(), target_gradient),
               "backward target gradient correct");
    test.check(test::compare(layer.get_source_gradient(), source_gradient),
               "backward source gradient correct");

    // check update
    const auto freeze = layer;
    layer.load(path + "_updated");

    test.check(
        test::compare(freeze.get_mha_1().get_query_linear().get_weights(),
                      layer.get_mha_1().get_query_linear().get_weights()),
        "first mha query_linear weights update correct");
    test.check(test::compare(freeze.get_mha_1().get_key_linear().get_weights(),
                             layer.get_mha_1().get_key_linear().get_weights()),
               "first mha key_linear bias update correct");
    test.check(
        test::compare(freeze.get_mha_1().get_value_linear().get_weights(),
                      layer.get_mha_1().get_value_linear().get_weights()),
        "first mha value_linear weights update correct");
    test.check(
        test::compare(freeze.get_mha_1().get_output_linear().get_weights(),
                      layer.get_mha_1().get_output_linear().get_weights()),
        "first mha output_linear weights update correct");
    test.check(test::compare(freeze.get_mha_1().get_output_linear().get_bias(),
                             layer.get_mha_1().get_output_linear().get_bias()),
               "first mha output_linear bias update correct");
    test.check(test::compare(freeze.get_mha_1_norm().get_gamma(),
                             layer.get_mha_1_norm().get_gamma()),
               "first mha norm gamma update correct");
    test.check(test::compare(freeze.get_mha_1_norm().get_beta(),
                             layer.get_mha_1_norm().get_beta()),
               "first mha norm beta update correct");
    test.check(
        test::compare(freeze.get_mha_2().get_query_linear().get_weights(),
                      layer.get_mha_2().get_query_linear().get_weights()),
        "second mha query_linear weights update correct");
    test.check(test::compare(freeze.get_mha_2().get_key_linear().get_weights(),
                             layer.get_mha_2().get_key_linear().get_weights()),
               "second mha key_linear bias update correct");
    test.check(
        test::compare(freeze.get_mha_2().get_value_linear().get_weights(),
                      layer.get_mha_2().get_value_linear().get_weights()),
        "second mha value_linear weights update correct");
    test.check(
        test::compare(freeze.get_mha_2().get_output_linear().get_weights(),
                      layer.get_mha_2().get_output_linear().get_weights()),
        "second mha output_linear weights update correct");
    test.check(test::compare(freeze.get_mha_2().get_output_linear().get_bias(),
                             layer.get_mha_2().get_output_linear().get_bias()),
               "second mha output_linear bias update correct");
    test.check(test::compare(freeze.get_mha_2_norm().get_gamma(),
                             layer.get_mha_2_norm().get_gamma()),
               "second mha norm gamma update correct");
    test.check(test::compare(freeze.get_mha_2_norm().get_beta(),
                             layer.get_mha_2_norm().get_beta()),
               "second mha norm beta update correct");
    test.check(test::compare(freeze.get_pwff().get_dense_1().get_weights(),
                             layer.get_pwff().get_dense_1().get_weights()),
               "pwff dense_1 weights update correct");
    test.check(test::compare(freeze.get_pwff().get_dense_1().get_bias(),
                             layer.get_pwff().get_dense_1().get_bias()),
               "pwff dense_1 bias update correct");
    test.check(test::compare(freeze.get_pwff().get_dense_2().get_weights(),
                             layer.get_pwff().get_dense_2().get_weights()),
               "pwff dense_2 weights update correct");
    test.check(test::compare(freeze.get_pwff().get_dense_2().get_bias(),
                             layer.get_pwff().get_dense_2().get_bias()),
               "pwff dense_2 bias update correct");
    test.check(test::compare(freeze.get_pwff_norm().get_gamma(),
                             layer.get_pwff_norm().get_gamma()),
               "pwff norm gamma update correct");
    test.check(test::compare(freeze.get_pwff_norm().get_beta(),
                             layer.get_pwff_norm().get_beta()),
               "pwff norm beta update correct");
}

/**
 * @brief This function tests the Encoder layer using random data.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_encoder(test::Test &test) {
    test.component("Encoder");
    std::string path = test.data_path("layers/encoder");

    core::Device device;

    auto learning_rate =
        core::Host_Tensor<core::data_type>(path + "_learning_rate")[0];
    std::unordered_map<std::string, core::data_type> optimizer_kw{
        {"learning_rate", learning_rate}};

    auto dropout = core::Host_Tensor<core::data_type>(path + "_dropout")[0];
    if (dropout < epsilon) test.comment("We test here without dropout!");

    auto num_heads =
        core::Host_Tensor<core::index_type>(path + "_num_heads")[0];
    auto num_layers =
        core::Host_Tensor<core::index_type>(path + "_num_layers")[0];
    auto num_embeddings =
        core::Host_Tensor<core::label_type>(path + "_num_embeddings")[0];
    auto max_batchsize =
        core::Host_Tensor<core::index_type>(path + "_max_batchsize")[0];
    auto hidden_dim =
        core::Host_Tensor<core::index_type>(path + "_hidden_dim")[0];

    core::Device_Tensor<core::label_type> input(path + "_input_1");
    core::Device_Tensor<core::mask_type> mask(path + "_mask");
    core::Host_Tensor<core::data_type> output(path + "_output_1");
    core::Host_Tensor<core::data_type> first_layer_input(
        path + "_first_layer_input_1");
    core::Device_Tensor<core::data_type> error(path + "_error");
    core::Host_Tensor<core::data_type> first_layer_input_gradient(
        path + "_first_layer_input_gradient");

    auto hidden_shape = output.shape_vec(core::NO_BATCHSIZE);
    hidden_shape.back() = hidden_dim;

    layers::Encoder<core::label_type, core::data_type, optimizers::SGD> layer(
        device, input.shape_vec(core::NO_BATCHSIZE),
        output.shape_vec(core::NO_BATCHSIZE), hidden_shape, num_layers,
        num_embeddings, optimizer_kw, num_heads, dropout, max_batchsize);

    layer.load(path);

    // forward and backward pass
    test.check(test::compare(layer.forward(input, mask), output),
               "forward correct");
    test.check(test::compare(layer.get_positionalencoding().get_output(),
                             first_layer_input),
               "forward first layer input correct");
    layer.backward(error);
    test.check(test::compare(layer.get_layer(0).get_input_gradient(),
                             first_layer_input_gradient),
               "backward first layer input gradient correct");

    // check second step forward
    input.load(path + "_input_2");
    output.load(path + "_output_2");
    first_layer_input.load(path + "_first_layer_input_2");
    test.check(test::compare<core::data_type>(layer.forward(input, mask),
                                              output, 1000),
               "forward second step correct");
    test.check(test::compare<core::data_type>(
                   layer.get_positionalencoding().get_output(),
                   first_layer_input, 1000),
               "forward second step first layer input correct");
}

/**
 * @brief This function tests the Decoder layer using random data.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_decoder(test::Test &test) {
    test.component("Decoder");
    std::string path = test.data_path("layers/decoder");

    core::Device device;

    auto learning_rate =
        core::Host_Tensor<core::data_type>(path + "_learning_rate")[0];
    std::unordered_map<std::string, core::data_type> optimizer_kw{
        {"learning_rate", learning_rate}};

    auto dropout = core::Host_Tensor<core::data_type>(path + "_dropout")[0];
    if (dropout < epsilon) test.comment("We test here without dropout!");

    auto num_heads =
        core::Host_Tensor<core::index_type>(path + "_num_heads")[0];
    auto num_layers =
        core::Host_Tensor<core::index_type>(path + "_num_layers")[0];
    auto num_embeddings =
        core::Host_Tensor<core::label_type>(path + "_num_embeddings")[0];
    auto max_batchsize =
        core::Host_Tensor<core::index_type>(path + "_max_batchsize")[0];
    auto hidden_dim =
        core::Host_Tensor<core::index_type>(path + "_hidden_dim")[0];

    core::Device_Tensor<core::label_type> target(path + "_target_1");
    core::Device_Tensor<core::mask_type> target_mask(path + "_target_mask");
    core::Device_Tensor<core::data_type> source(path + "_source_1");
    core::Device_Tensor<core::mask_type> source_mask(path + "_source_mask");
    core::Host_Tensor<core::data_type> output(path + "_output_1");
    core::Host_Tensor<core::data_type> first_layer_input(
        path + "_first_layer_input_1");
    core::Host_Tensor<core::data_type> attention(path + "_attention_1");
    core::Device_Tensor<core::data_type> error(path + "_error");
    core::Host_Tensor<core::data_type> source_gradient(path +
                                                       "_source_gradient");
    core::Host_Tensor<core::data_type> first_layer_input_gradient(
        path + "_first_layer_input_gradient");

    auto hidden_shape = output.shape_vec(core::NO_BATCHSIZE);
    hidden_shape.back() = hidden_dim;

    layers::Decoder<core::label_type, core::data_type, optimizers::SGD> layer(
        device, target.shape_vec(core::NO_BATCHSIZE),
        source.shape_vec(core::NO_BATCHSIZE), hidden_shape, num_layers,
        num_embeddings, optimizer_kw, num_heads, dropout, max_batchsize);

    layer.load(path);

    // forward and backward pass
    test.check(
        test::compare(layer.forward(target, target_mask, source, source_mask),
                      output),
        "forward correct");
    test.check(test::compare(layer.get_positionalencoding().get_output(),
                             first_layer_input),
               "forward first layer input correct");
    test.check(test::compare(layer.get_attention(), attention),
               "attention correct");

    layer.backward(error);
    test.check(test::compare(layer.get_source_gradient(), source_gradient),
               "backward source gradient correct");
    test.check(test::compare(layer.get_layer(0).get_target_gradient(),
                             first_layer_input_gradient),
               "backward first layer input gradient correct");

    // check second step forward
    target.load(path + "_target_2");
    source.load(path + "_source_2");
    output.load(path + "_output_2");
    first_layer_input.load(path + "_first_layer_input_2");
    attention.load(path + "_attention_2");
    test.check(test::compare<core::data_type>(
                   layer.forward(target, target_mask, source, source_mask),
                   output, 1000),
               "forward second step correct");
    test.check(test::compare<core::data_type>(
                   layer.get_positionalencoding().get_output(),
                   first_layer_input, 1000),
               "forward first layer input second step correct");
    test.check(
        test::compare<core::data_type>(layer.get_attention(), attention, 100),
        "attention second step correct");
}

/**
 * @brief This function tests the Transformer layer using random data.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_transformer(test::Test &test) {
    test.component("Transformer");
    std::string path = test.data_path("layers/transformer");

    core::Device device;

    auto learning_rate =
        core::Host_Tensor<core::data_type>(path + "_learning_rate")[0];
    std::unordered_map<std::string, core::data_type> optimizer_kw{
        {"learning_rate", learning_rate}};

    auto dropout = core::Host_Tensor<core::data_type>(path + "_dropout")[0];
    if (dropout < epsilon) test.comment("We test here without dropout!");

    auto num_heads =
        core::Host_Tensor<core::index_type>(path + "_num_heads")[0];
    auto num_layers =
        core::Host_Tensor<core::index_type>(path + "_num_layers")[0];
    auto num_source_embeddings =
        core::Host_Tensor<core::label_type>(path + "_num_source_embeddings")[0];
    auto num_target_embeddings =
        core::Host_Tensor<core::label_type>(path + "_num_target_embeddings")[0];
    auto max_batchsize =
        core::Host_Tensor<core::index_type>(path + "_max_batchsize")[0];
    auto hidden_dim =
        core::Host_Tensor<core::index_type>(path + "_hidden_dim")[0];
    auto ignore_index =
        core::Host_Tensor<core::label_type>(path + "_ignore_index")[0];
    auto embedding_dim =
        core::Host_Tensor<core::index_type>(path + "_embedding_dim")[0];

    core::Device_Tensor<core::label_type> source(path + "_source_1");
    core::Device_Tensor<core::label_type> target(path + "_target_1");
    core::Host_Tensor<core::data_type> output(path + "_output_1");
    core::Host_Tensor<core::data_type> attention(path + "_attention_1");
    core::Host_Tensor<core::mask_type> source_mask(path + "_source_mask_1");
    core::Host_Tensor<core::mask_type> target_mask(path + "_target_mask_1");
    core::Host_Tensor<core::data_type> encoder_output(path +
                                                      "_encoder_output_1");
    core::Device_Tensor<core::data_type> error(path + "_error");
    core::Host_Tensor<core::data_type> encoder_first_layer_input_gradient(
        path + "_encoder_first_layer_input_gradient");
    core::Host_Tensor<core::data_type> decoder_source_gradient(
        path + "_decoder_source_gradient");
    core::Host_Tensor<core::data_type> decoder_first_layer_input_gradient(
        path + "_decoder_first_layer_input_gradient");

    layers::Transformer<core::label_type, core::data_type, optimizers::SGD>
        layer(device, source.shape_vec(core::NO_BATCHSIZE),
              target.shape_vec(core::NO_BATCHSIZE), embedding_dim, hidden_dim,
              num_layers, num_layers, num_source_embeddings,
              num_target_embeddings, optimizer_kw, num_heads, dropout,
              ignore_index, max_batchsize);

    layer.load(path);

    // forward and backward pass
    test.check(test::compare(layer.forward(source, target), output),
               "forward correct");
    test.check(test::compare(layer.get_attention(), attention),
               "attention correct");
    test.check(test::compare(layer.get_source_mask(), source_mask),
               "source mask correct");
    test.check(test::compare(layer.get_target_mask(), target_mask),
               "target mask correct");
    test.check(test::compare(layer.get_encoder().get_output(), encoder_output),
               "encoder output correct");

    layer.backward(error);
    test.check(
        test::compare(layer.get_decoder().get_layer(0).get_target_gradient(),
                      decoder_first_layer_input_gradient),
        "decoder first layer input gradient correct");
    test.check(test::compare(layer.get_decoder().get_source_gradient(),
                             decoder_source_gradient),
               "decoder source gradient correct");
    test.check(
        test::compare(layer.get_encoder().get_layer(0).get_input_gradient(),
                      encoder_first_layer_input_gradient),
        "encoder first layer input gradient correct");

    // check second step forward
    source.load(path + "_source_2");
    target.load(path + "_target_2");
    output.load(path + "_output_2");
    attention.load(path + "_attention_2");
    source_mask.load(path + "_source_mask_2");
    target_mask.load(path + "_target_mask_2");
    encoder_output.load(path + "_encoder_output_2");

    test.check(test::compare<core::data_type>(layer.forward(source, target),
                                              output, 1000),
               "forward second step correct");
    test.check(
        test::compare<core::data_type>(layer.get_attention(), attention, 1000),
        "attention second step correct");
    test.check(test::compare(layer.get_source_mask(), source_mask),
               "source mask second step correct");
    test.check(test::compare(layer.get_target_mask(), target_mask),
               "target mask second step correct");
    test.check(test::compare<core::data_type>(layer.get_encoder().get_output(),
                                              encoder_output, 1000),
               "encoder output second step correct");
}

/**
 * @brief Launches all tests defined in this file.
 *
 * This function initializes the test framework using command line arguments
 * and runs a series of tests for the layers defined in the `layers` namespace.
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
    test.series("Layers");
    test_dense_initialization(test);
    test_dense_without_bias(test);
    test_dense_with_bias(test);
    test_reshape(test);
    test_upscale(test);
    test_dropout(test);
    test_layernorm(test);
    test_embedding(test);
    test_positionwisefeedforward(test);
    test_positionalencoding(test);
    test_multiheadattention(test);
    test_encoderlayer(test);
    test_decoderlayer(test);
    test_encoder(test);
    test_decoder(test);
    test_transformer(test);
    return test.eval();
}
