/**
 * @file 01_core.cu
 * @brief Tests the tensor classes defined in the `core` namespace.
 */

#include <utility>

#include "../src/core.cuh"
#include "../src/test.cuh"

/**
 * @brief This function tests the constructor of the Host_Tensor class for
 * various scenarios.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_host_tensor_constructor(test::Test &test) {
    test.component("Host Tensor - Constructor");

    // empty
    core::Host_Tensor<int> empty;
    test.check(empty.rank() == 0, "empty correct rank");
    test.check(empty.shape() == nullptr, "empty correct shape");
    test.check(empty.batchsize() == 0, "empty correct batchsize");
    test.check(empty.size() == 0, "empty correct number of elements");
    test.check(empty.sample_size() == 0, "empty correct sample_size");
    test.check(empty.data() == nullptr, "empty correct initialization");

    test.comment("empty string method: " + empty.str());

    // scalar
    core::Host_Tensor<int> scalar({});
    test.check(scalar.rank() == 0, "scalar correct rank");
    test.check(scalar.shape() == nullptr, "scalar correct shape");
    test.check(scalar.batchsize() == 0, "scalar correct batchsize");
    test.check(scalar.sample_size() == 1, "scalar correct sample_size");
    test.check(scalar.size() == 1, "scalar correct number of elements");
    scalar[0] = 1;
    test.check(scalar[0] == 1, "scalar correct access");

    test.comment("scalar string method: " + scalar.str());

    // multi-dimensional tensor
    core::Host_Tensor<int> tensor({3, 5, 8});
    test.check(tensor.rank() == 3, "correct rank");
    test.check(
        tensor.shape(0) == 3 && tensor.shape(1) == 5 && tensor.shape(2) == 8,
        "correct shape");
    test.check(tensor.batchsize() == 3, "correct batchsize");
    test.check(tensor.sample_size() == 5 * 8, "correct sample_size");
    test.check(tensor.size() == 3 * 5 * 8, "correct number of elements");

    for (int idx = 0; idx < tensor.size(); ++idx) tensor[idx] = idx;

    test.comment("tensor string method: " + tensor.str());
}

/**
 * @brief Tests the copy and move constructor, assignment operator, and
 * `copy_from` function of the Host_Tensor class.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_host_tensor_copy_and_move(test::Test &test) {
    test.component("Host Tensor - Copy and Move");

    core::Host_Tensor<int> tensor({3, 5, 8});
    for (size_t i = 0; i < tensor.size(); ++i) tensor[i] = i;

    // copy assignment
    core::Host_Tensor<int> tensor_copy = tensor;
    test.check(test::compare(tensor, tensor_copy), "copy-assignment");
    tensor_copy[0] = 1;
    test.check(tensor[0] == 0, "correct state after copy-assignment");
    tensor_copy = tensor;

    // copy constructor
    core::Host_Tensor<int> tensor_copy_constructor(tensor);
    test.check(test::compare(tensor, tensor_copy_constructor),
               "copy-constructor");
    tensor_copy_constructor[0] = 1;
    test.check(tensor[0] == 0, "correct state after copy-constructor");

    // copy_from function
    core::Host_Tensor<int> tensor_copy_from_fkt;
    tensor_copy_from_fkt.copy_from(tensor);
    test.check(test::compare(tensor, tensor_copy_from_fkt),
               "copy_from function");
    tensor_copy_from_fkt[0] = 1;
    test.check(tensor[0] == 0, "correct state after copy_from function");

    // asynchronous copy_from function
    core::Host_Tensor<int> tensor_copy_from_fkt_async;
    tensor_copy_from_fkt_async.copy_from(tensor, core::default_stream);
    core::checkCuda(cudaDeviceSynchronize());
    test.check(test::compare(tensor, tensor_copy_from_fkt_async),
               "asynchronous copy_from function");
    tensor_copy_from_fkt_async[0] = 1;
    test.check(tensor[0] == 0,
               "correct state after asynchronous copy_from function");

    // move assignment
    core::Host_Tensor<int> tensor_move;
    tensor_move = std::move(tensor);
    test.check(test::compare(tensor, core::Host_Tensor<int>()),
               "correct state after move-assignment");
    test.check(test::compare(tensor_copy, tensor_move),
               "move-assignment succeeded");

    // move constructor
    tensor = tensor_copy;
    core::Host_Tensor<int> tensor_move_constructor(std::move(tensor));
    test.check(test::compare(tensor, core::Host_Tensor<int>()),
               "correct state after move-constructor");
    test.check(test::compare(tensor_copy, tensor_move_constructor),
               "move-constructor succeeded");
}

/**
 * @brief Tests the file IO functionality of the Host_Tensor class.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_host_tensor_file_io(test::Test &test) {
    test.component("Host Tensor - File IO");

    // read a tensor from disk
    core::Host_Tensor<int> tensor_01(test.data_path("core/tensor_01"));
    test.check(tensor_01.rank() == 2, "tensor_01 correct rank");
    test.check(tensor_01.shape(0) == 2 && tensor_01.shape(1) == 3,
               "tensor_01 correct shape");
    test.check(tensor_01.batchsize() == 2, "tensor_01 correct batchsize");
    test.check(tensor_01.sample_size() == 3, "tensor_01 correct sample_size");
    test.check(tensor_01.size() == 2 * 3,
               "tensor_01 correct number of elements");

    for (int i = 0; i < tensor_01.size(); ++i)
        test.check(tensor_01[i] == i, "tensor_01 correct entry");

    // write a tensor to disk
    tensor_01.save(test.temporary_path("tensor"));
    core::Host_Tensor<int> tensor_02(test.temporary_path("tensor"));
    test.check(test::compare(tensor_01, tensor_02),
               "tensor_02 read/write correct");

    // read a scalar from disk
    core::Host_Tensor<int> scalar_01(test.data_path("core/scalar_01"));
    test.check(scalar_01.rank() == 0, "scalar_01 correct rank");
    test.check(scalar_01.size() == 1, "scalar_01 correct number of elements");
    test.check(scalar_01.batchsize() == 0, "scalar_01 correct batchsize");
    test.check(scalar_01.sample_size() == 1, "scalar_01 correct sample_size");
    test.check(scalar_01[0] == 42, "scalar_01 correct entry");

    // write a scalar to disk
    scalar_01.save(test.temporary_path("scalar"));
    core::Host_Tensor<int> scalar_02(test.temporary_path("scalar"));
    test.check(test::compare(scalar_01, scalar_02),
               "scalar_02 read/write correct");
}

/**
 * @brief Tests the reshape and rebatchsize functionality of the Host_Tensor
 * class.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_host_reshape(test::Test &test) {
    test.component("Host Tensor - Reshape");

    core::Host_Tensor<int> tensor({3, 5, 8});
    for (int idx = 0; idx < tensor.size(); ++idx) tensor[idx] = idx;

    auto tensor_copy = tensor;

    // reshape with same shape
    tensor.reshape({3, 5, 8});
    test.check(test::compare(tensor, tensor_copy),
               "correct state after reshape with same shape");

    // rebatchsize with same batchsize
    tensor.rebatchsize(3);
    test.check(test::compare(tensor, tensor_copy),
               "correct state after rebatchsize with same batchsize");

    // reshape with same size
    tensor.reshape({3, 5 * 8});
    test.check(tensor.rank() == 2, "correct rank after reshape with same size");
    test.check(tensor.shape(0) == 3 && tensor.shape(1) == 5 * 8,
               "correct shape after reshape with same size");
    test.check(tensor.batchsize() == 3,
               "correct batchsize after reshape with same size");
    test.check(tensor.sample_size() == 5 * 8,
               "correct sample_size after reshape with same size");
    test.check(tensor.size() == 3 * 5 * 8,
               "correct number of elements after reshape with same size");

    tensor.reshape({3, 5, 8});
    test.check(test::compare(tensor, tensor_copy),
               "correct state after reshape to original shape");

    tensor = tensor_copy;

    // reshape to a different size
    tensor.reshape({1});
    test.check(tensor.rank() == 1, "correct rank after reshape");
    test.check(tensor.shape(0) == 1, "correct shape after reshape");
    test.check(tensor.batchsize() == 1, "correct batchsize after reshape");
    test.check(tensor.sample_size() == 1, "correct sample_size after reshape");
    test.check(tensor.size() == 1, "correct number of elements after reshape");
    tensor[0] = 0;
    test.check(tensor[0] == 0, "correct state after reshape");

    tensor = tensor_copy;

    // rebatchsize to different batchsize
    tensor.rebatchsize(7);
    test.check(tensor.rank() == 3, "correct rank after rebatchsize");
    test.check(
        tensor.shape(0) == 7 && tensor.shape(1) == 5 && tensor.shape(2) == 8,
        "correct shape after rebatchsize");
    test.check(tensor.batchsize() == 7, "correct batchsize after rebatchsize");
    test.check(tensor.sample_size() == 5 * 8,
               "correct sample_size after rebatchsize");
    test.check(tensor.size() == 7 * 5 * 8,
               "correct number of elements after rebatchsize");
    tensor[0] = 0;
    test.check(tensor[0] == 0, "correct state after rebatchsize");

    // rebatchsize with NO_ALLOC
    tensor = core::Host_Tensor<int>(core::NO_ALLOC, {5, 8});
    tensor.rebatchsize(7);
    test.check(tensor.rank() == 3, "correct rank after NO_ALLOC constructor");
    test.check(
        tensor.shape(0) == 7 && tensor.shape(1) == 5 && tensor.shape(2) == 8,
        "correct shape after NO_ALLOC constructor");
    test.check(tensor.batchsize() == 7,
               "correct batchsize after NO_ALLOC constructor");
    test.check(tensor.sample_size() == 5 * 8,
               "correct sample_size after NO_ALLOC constructor");
    test.check(tensor.size() == 7 * 5 * 8,
               "correct number of elements after NO_ALLOC constructor");
    tensor[0] = 0;
    test.check(tensor[0] == 0, "correct state after NO_ALLOC constructor");
}

/**
 * @brief This kernel sets the first entry of the `tensor` to 1.
 *
 * @see test_device_tensor_constructor
 * @see test_device_tensor_copy_and_move
 * @see test_device_reshape
 */
__global__ void set_value(core::Kernel_Tensor<int> tensor) { tensor[0] = 1; }

/**
 * @brief This function tests the constructor of the Device_Tensor class for
 * various scenarios.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_device_tensor_constructor(test::Test &test) {
    test.component("Device Tensor - Constructor");

    // empty
    core::Device_Tensor<int> empty;
    test.check(empty.rank() == 0, "empty correct rank");
    test.check(empty.shape() == nullptr, "empty correct shape");
    test.check(empty.batchsize() == 0, "empty correct batchsize");
    test.check(empty.sample_size() == 0, "empty correct sample_size");
    test.check(empty.size() == 0, "empty correct number of elements");
    test.check(empty.data() == nullptr, "emtpy correct initialization");

    // scalar
    core::Device_Tensor<int> scalar({});
    core::Kernel_Tensor<int> kernel_scalar = scalar;
    test.check(scalar.rank() == 0, "scalar correct rank");
    test.check(scalar.shape() == nullptr, "scalar correct shape");
    test.check(scalar.batchsize() == 0, "scalar correct batchsize");
    test.check(scalar.sample_size() == 1, "scalar correct sample_size");
    test.check(scalar.size() == 1, "scalar correct number of elements");

    set_value<<<1, 1>>>(scalar);
    core::checkCuda(cudaGetLastError());
    core::checkCuda(cudaDeviceSynchronize());
    auto scalar_access_test = [kernel_scalar] __device__(bool *const ret) {
        *ret = kernel_scalar[0] == 1;
    };
    test.check(test::test_kernel(scalar_access_test), "scalar correct access");

    // multi-dimensional tensor
    core::Device_Tensor<int> tensor({3, 5, 8});
    core::Kernel_Tensor<int> kernel_tensor = tensor;
    test.check(tensor.rank() == 3, "tensor correct rank");
    auto tensor_shape_test = [kernel_tensor] __device__(bool *const ret) {
        *ret = kernel_tensor.shape(0) == 3 && kernel_tensor.shape(1) == 5 &&
               kernel_tensor.shape(2) == 8;
    };
    test.check(test::test_kernel(tensor_shape_test), "tensor correct shape");

    test.check(tensor.batchsize() == 3, "tensor correct batchsize");
    test.check(tensor.sample_size() == 5 * 8, "tensor correct sample_size");
    test.check(tensor.size() == 3 * 5 * 8, "tensor correct number of elements");
}

/**
 * @brief Tests the copy and move constructor, assignment operator, and
 * `copy_from` function of the Device_Tensor class.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_device_tensor_copy_and_move(test::Test &test) {
    test.component("Device Tensor - Copy and Move");

    core::Host_Tensor<int> host_tensor_01(test.data_path("core/tensor_01"));
    test.comment(host_tensor_01);

    // copy_from host function
    core::Device_Tensor<int> tensor_01;
    tensor_01.copy_from(host_tensor_01);
    test.check(test::compare(host_tensor_01, tensor_01),
               "correct state after copy_from host function");

    // asynchronous copy_from host function
    core::Device_Tensor<int> tensor_01_async;
    tensor_01_async.copy_from(host_tensor_01, core::default_stream);
    core::checkCuda(cudaDeviceSynchronize());
    test.check(test::compare(host_tensor_01, tensor_01_async),
               "correct state after asynchronous copy_from host function");

    // copy_from device to host function
    core::Host_Tensor<int> host_tensor_01_copy;
    host_tensor_01_copy.copy_from(tensor_01);
    test.check(test::compare(host_tensor_01, host_tensor_01_copy),
               "correct state after copy_from device to host function");
    test.comment(host_tensor_01_copy);

    // asynchronous copy_from device to host function
    core::Host_Tensor<int> host_tensor_01_copy_async;
    host_tensor_01_copy_async.copy_from(tensor_01, core::default_stream);
    core::checkCuda(cudaDeviceSynchronize());
    test.check(
        test::compare(host_tensor_01, host_tensor_01_copy_async),
        "correct state after asynchronous copy_from device to host function");

    // copy assignment
    core::Device_Tensor<int> tensor_02 = tensor_01;
    test.check(test::compare(tensor_02, host_tensor_01),
               "correct state after copy-assignment");

    set_value<<<1, 1>>>(tensor_02);
    core::checkCuda(cudaGetLastError());
    core::checkCuda(cudaDeviceSynchronize());
    test.check(!test::compare(tensor_02, tensor_01),
               "correct state after copy-assignment");
    tensor_02 = tensor_01;

    // copy constructor
    core::Device_Tensor<int> tensor_03(tensor_01);
    test.check(test::compare(tensor_03, host_tensor_01),
               "correct state after copy-constructor");

    set_value<<<1, 1>>>(tensor_03);
    core::checkCuda(cudaGetLastError());
    core::checkCuda(cudaDeviceSynchronize());
    test.check(!test::compare(tensor_03, tensor_01),
               "correct state after copy-constructor");
    tensor_03 = tensor_01;

    // copy_from device function
    core::Device_Tensor<int> tensor_04;
    tensor_04.copy_from(tensor_01);
    test.check(test::compare(tensor_04, host_tensor_01),
               "correct state after copy_from device function");

    set_value<<<1, 1>>>(tensor_04);
    core::checkCuda(cudaGetLastError());
    core::checkCuda(cudaDeviceSynchronize());
    test.check(!test::compare(tensor_04, tensor_01),
               "correct state after copy_from device function");
    tensor_04 = tensor_01;

    // asynchronous copy_from device function
    core::Device_Tensor<int> tensor_05;
    tensor_05.copy_from(tensor_01, core::default_stream);
    core::checkCuda(cudaDeviceSynchronize());
    test.check(test::compare(tensor_05, host_tensor_01),
               "correct state after asynchronous copy_from device function");

    set_value<<<1, 1>>>(tensor_05);
    core::checkCuda(cudaGetLastError());
    core::checkCuda(cudaDeviceSynchronize());
    test.check(!test::compare(tensor_05, tensor_01),
               "correct state after asynchronous copy_from device function");
    tensor_05 = tensor_01;

    // move assignment
    core::Device_Tensor<int> tensor_move;
    tensor_move = std::move(tensor_01);
    test.check(test::compare(tensor_01, core::Device_Tensor<int>()),
               "correct state after move-assignment");
    test.check(test::compare(tensor_02, tensor_move),
               "move-assignment succeeded");

    core::Device_Tensor<int> tensor_move_constructor(std::move(tensor_02));
    test.check(test::compare(tensor_02, core::Device_Tensor<int>()),
               "correct state after move-constructor");
    test.check(test::compare(tensor_03, tensor_move_constructor),
               "move-constructor succeeded");
}

/**
 * @brief Tests the file IO functionality of the Device_Tensor class.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_device_tensor_file_io(test::Test &test) {
    test.component("Device Tensor - File IO");

    // read a tensor from disk
    core::Host_Tensor<int> host_tensor_01(test.data_path("core/tensor_01"));
    core::Device_Tensor<int> device_tensor_01(test.data_path("core/tensor_01"));
    test.check(test::compare(host_tensor_01, device_tensor_01),
               "tensor_01 read correct!");

    // write a tensor to disk
    device_tensor_01.save(test.temporary_path("tensor"));
    core::Device_Tensor<int> device_tensor_02(test.temporary_path("tensor"));
    test.check(test::compare(device_tensor_01, device_tensor_02),
               "tensor_02 write correct");

    // asynchronous read a scalar from disk
    core::Host_Tensor<int> host_scalar_01(test.data_path("core/scalar_01"));
    core::Device_Tensor<int> device_scalar_01(test.data_path("core/scalar_01"),
                                              core::default_stream);
    core::checkCuda(cudaDeviceSynchronize());
    test.check(test::compare(host_scalar_01, device_scalar_01),
               "scalar_01 read correct!");

    // read a scalar from disk
    device_scalar_01.save(test.temporary_path("scalar"));
    core::Device_Tensor<int> device_scalar_02(test.temporary_path("scalar"));
    test.check(test::compare(device_scalar_01, device_scalar_02),
               "scalar_02 write correct");
}

/**
 * @brief Tests the reshape and rebatchsize functionality of the Device_Tensor
 * class.
 *
 * @param test The test framework used for output formatting, test counting, and
 * path manipulation.
 */
void test_device_reshape(test::Test &test) {
    test.component("Device Tensor - Reshape");

    core::Host_Tensor<int> host_tensor({3, 5, 8});
    for (int idx = 0; idx < host_tensor.size(); ++idx) host_tensor[idx] = idx;

    auto tensor = core::Device_Tensor(host_tensor);
    auto tensor_copy = tensor;

    // reshape with same shape
    tensor.reshape({3, 5, 8});
    test.check(test::compare(tensor, tensor_copy),
               "correct state after reshape with same shape");

    // rebatchsize with same batchsize
    tensor.rebatchsize(3);
    test.check(test::compare(tensor, tensor_copy),
               "correct state after rebatchsize with same batchsize");

    // reshape with same size
    tensor.reshape({3, 5 * 8});
    host_tensor.reshape({3, 5 * 8});
    test.check(test::compare(tensor, host_tensor),
               "correct state after reshape with same size");

    tensor.reshape({3, 5, 8});
    test.check(test::compare(tensor, tensor_copy),
               "correct state after reshape with original shape");

    tensor = tensor_copy;

    // reshape to a different size
    tensor.reshape({1});
    test.check(tensor.rank() == 1, "correct rank after reshape");
    auto shape = tensor.shape_vec();
    test.check(shape[0] == 1, "correct shape after reshape");
    test.check(tensor.batchsize() == 1, "correct batchsize after reshape");
    test.check(tensor.sample_size() == 1, "correct sample_size after reshape");
    test.check(tensor.size() == 1, "correct number of elements after reshape");

    set_value<<<1, 1>>>(tensor);
    core::checkCuda(cudaGetLastError());
    core::checkCuda(cudaDeviceSynchronize());
    core::Kernel_Tensor<int> kernel_tensor = tensor;
    auto reshape_access_test = [kernel_tensor] __device__(bool *const ret) {
        *ret = kernel_tensor[0] == 1;
    };
    test.check(test::test_kernel(reshape_access_test),
               "correct state after reshape");

    tensor = tensor_copy;

    // rebatchsize to different batchsize
    tensor.rebatchsize(7);
    test.check(tensor.rank() == 3, "correct rank after rebatchsize");
    shape = tensor.shape_vec();
    test.check(shape[0] == 7 && shape[1] == 5 && shape[2] == 8,
               "correct shape after rebatchsize");
    test.check(tensor.batchsize() == 7, "correct batchsize after rebatchsize");
    test.check(tensor.sample_size() == 5 * 8,
               "correct sample_size after rebatchsize");
    test.check(tensor.size() == 7 * 5 * 8,
               "correct number of elements after rebatchsize");

    // asynchronous rebatchsize
    tensor = tensor_copy;
    tensor.rebatchsize(7, core::default_stream);
    core::checkCuda(cudaDeviceSynchronize());
    test.check(tensor.rank() == 3,
               "correct rank after asynchronous rebatchsize");
    shape = tensor.shape_vec();
    test.check(shape[0] == 7 && shape[1] == 5 && shape[2] == 8,
               "correct shape after asynchronous rebatchsize");
    test.check(tensor.batchsize() == 7,
               "correct batchsize after asynchronous rebatchsize");
    test.check(tensor.sample_size() == 5 * 8,
               "correct sample_size after asynchronous rebatchsize");
    test.check(tensor.size() == 7 * 5 * 8,
               "correct number of elements after asynchronous rebatchsize");

    set_value<<<1, 1>>>(tensor);
    core::checkCuda(cudaGetLastError());
    core::checkCuda(cudaDeviceSynchronize());
    kernel_tensor = tensor;
    auto rebatchsize_access_test = [kernel_tensor] __device__(bool *const ret) {
        *ret = kernel_tensor[0] == 1;
    };
    test.check(test::test_kernel(rebatchsize_access_test),
               "correct state after rebatchsize");

    // rebatchsize on NO_ALLOC tensor
    tensor = core::Device_Tensor<int>(core::NO_ALLOC, {5, 8});
    tensor.rebatchsize(7);
    test.check(tensor.rank() == 3, "correct rank after NO_ALLOC constructor");
    shape = tensor.shape_vec();
    test.check(shape[0] == 7 && shape[1] == 5 && shape[2] == 8,
               "correct shape after NO_ALLOC constructor");
    test.check(tensor.batchsize() == 7,
               "correct batchsize after NO_ALLOC constructor");
    test.check(tensor.sample_size() == 5 * 8,
               "correct sample_size after NO_ALLOC constructor");
    test.check(tensor.size() == 7 * 5 * 8,
               "correct number of elements after NO_ALLOC constructor");

    set_value<<<1, 1>>>(tensor);
    core::checkCuda(cudaGetLastError());
    core::checkCuda(cudaDeviceSynchronize());
    kernel_tensor = tensor;
    auto no_alloc_access_test = [kernel_tensor] __device__(bool *const ret) {
        *ret = kernel_tensor[0] == 1;
    };
    test.check(test::test_kernel(no_alloc_access_test),
               "correct state after NO_ALLOC constructor");
}

/**
 * @brief Launches all tests defined in this file.
 *
 * This function initializes the test framework using command line arguments
 * and runs a series of tests for the tensor classes defined in the `core`
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
    test.series("Core");
    test_host_tensor_constructor(test);
    test_host_tensor_copy_and_move(test);
    test_host_tensor_file_io(test);
    test_host_reshape(test);
    test_device_tensor_constructor(test);
    test_device_tensor_copy_and_move(test);
    test_device_tensor_file_io(test);
    test_device_reshape(test);
    return test.eval();
}
