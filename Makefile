.PHONY: clean test compile_tests data run exclude_all_data_generation_scripts \
	include_all_data_generation_scripts exclude_all_tests include_all_tests \
	compile_file transformer weights_update dropout matrix_multiplication \
	transformer_overview transformer_scaling atomic stream mnist \
	mnist_extended plot

# Define initial parameters.
CUDA_FLAGS := -O3 -std=c++20 -Wall -Wextra -pedantic -cuda -lcublas -lcurand \
	-fmax-errors=10
BIN_DIR := bin
SRC_DIR := src
TEST_DIR := test
MAIN_DIR := main
DATA_DIR := data
LOG_DIR := log
DATA_GENERATION_SCRIPTS_DIR := data_generation_scripts
SHELL_SCRIPTS_DIR := shell_scripts
MEMCHECK := 0
CONTINUE_ON_FAIL := 0
NCU = 1

# If `DEBUG=1` add `--DDEBUG` to the `CUDA_FLAGS`.
DEBUG ?=
ifeq ($(DEBUG), 1)
CUDA_FLAGS += -DDEBUG
endif

# The `test` target runs the compiled test executables. Files starting with `_`
# get excluded.
#   - If `MEMCHECK` is set to 1, it uses compute-sanitizer to check for memory
#     leaks.
#   - If `MEMCHECK` is set to 0, it runs the executables directly without
#     compute-sanitizer.
#   - If `CONTINUE_ON_FAIL` is set to 1, it will continue running even if a test
#     fails.
#
# Usage:
#    `make test [MEMCHECK=0|1] [CONTINUE_ON_FAIL=0|1]`
#
# When running the tests two cases can occur:
#    - Case 1: If the return value is negative, print an error message and abort
#      the operation.
#    - Case 2: If the return value is positive, add it to the total number of
#      failed tests.
#
# For more information see `SHELL_SCRIPTS_DIR/run_tests.sh`.
test: data compile_tests
	@./$(SHELL_SCRIPTS_DIR)/run_tests.sh $(BIN_DIR) $(DATA_DIR) $(MEMCHECK) \
		$(CONTINUE_ON_FAIL)

# The `compile_tests` target compiles all the `*.cpp` and `*.cu` files in the
# test directory. Files starting with `_` get excluded. It places the compiled
# executables in the `BIN_DIR/test` directory.
#
# For more information see `SHELL_SCRIPTS_DIR/compile_tests.sh`.
compile_tests: CUDA_FLAGS += -DDEBUG
compile_tests:
	@./$(SHELL_SCRIPTS_DIR)/compile_tests.sh $(BIN_DIR) "$(CUDA_FLAGS)" \
		$(TEST_DIR)
	@echo ""

# The `data` target runs all the Python scripts in
# `DATA_GENERATION_SCRIPTS_DIR`. Files starting with `_` get excluded. It
# executes each script found and passes the `DATA_DIR` as a command line
# argument. If any script fails, the operation will abort.
#
# For more information see `SHELL_SCRIPTS_DIR/generate_data.sh`.
data:
	@./$(SHELL_SCRIPTS_DIR)/generate_data.sh $(DATA_DIR) \
		$(DATA_GENERATION_SCRIPTS_DIR)
	@echo ""

# The `run` target is a general rule to build an executable from a `*.cpp` or
# `*.cu` file. The `%` character is a wildcard that can match any string.
#
# Usage:
#    `make run FILE=path/to/some/cpp/or/cu/file ARGS="arg1 arg2"`
run: compile_file $(BIN_DIR)/$(notdir $(basename $(FILE)))
	$(BIN_DIR)/$(notdir $(basename $(FILE))) $(ARGS)

# The `compile_file` target compiles the `FILE`. This target is mainly used in
# conjunction with the `run` target.
compile_file: $(FILE)
	mkdir -p $(BIN_DIR)
	nvc++ $(CUDA_FLAGS) $< -o $(BIN_DIR)/$(notdir $(basename $(FILE)))

# The `exclude_all_data_generation_scripts` target renames all files starting
# with digits in `DATA_GENERATION_SCRIPTS_DIR` to start with `_`. Hence, they
# will not be executed in the `data` target.
#
# For more information see `SHELL_SCRIPTS_DIR/rename_files.sh`.
exclude_all_data_generation_scripts:
	@./$(SHELL_SCRIPTS_DIR)/rename_files.sh $(DATA_GENERATION_SCRIPTS_DIR) \
		exclude

# The `include_all_data_generation_scripts` target renames all files in
# `DATA_GENERATION_SCRIPTS_DIR` to remove the leading `_`, i.e., reversing the
# `exclude_all_data_generation_scripts` target.
#
# For more information see `SHELL_SCRIPTS_DIR/rename_files.sh`.
include_all_data_generation_scripts:
	@./$(SHELL_SCRIPTS_DIR)/rename_files.sh $(DATA_GENERATION_SCRIPTS_DIR) \
		include

# The `exclude_all_tests` target renames all files starting with digits in
# `TEST_DIR` to start with `_`. Hence, they will not be compiled in the
# `compile_tests` target.
#
# For more information see `SHELL_SCRIPTS_DIR/rename_files.sh`.
exclude_all_tests:
	@./$(SHELL_SCRIPTS_DIR)/rename_files.sh $(TEST_DIR) exclude

# The `include_all_tests` target renames all files in `TEST_DIR` to remove the
# leading `_`, i.e., reversing the `exclude_all_tests` target.
#
# For more information see `SHELL_SCRIPTS_DIR/rename_files.sh`.
include_all_tests:
	@./$(SHELL_SCRIPTS_DIR)/rename_files.sh $(TEST_DIR) include

# The `exclude_all` target is a short hand for executing the `exclude_all_tests`
# and `exclude_all_data_generation_scripts`.
exclude_all: exclude_all_data_generation_scripts exclude_all_tests

# The `include_all` target is a short hand for executing the `include_all_tests`
# and `include_all_data_generation_scripts`.
include_all: include_all_data_generation_scripts include_all_tests

# The `transformer` target compiles and runs `main/transformer.cu` with
# different configurations and benchmarks the code.
#   - If `DEBUG` is set to 1, only one single configuration is run.
#     Additionally, the code will be compiled with the `-DDEBUG` flag.
#   - If `NCU` is set to 0, the Nvidia Compute Utility kernel measurement will
#     be omitted.
#
# Usage:
#    `make transformer [DEBUG=0|1] [NCU=0|1]`
transformer:
	mkdir -p $(DATA_DIR)
	python $(DATA_GENERATION_SCRIPTS_DIR)/99_transformer.py $(DATA_DIR) \
		$(TMPDIR)

	mkdir -p $(BIN_DIR)
	nvc++ $(CUDA_FLAGS) $(MAIN_DIR)/transformer.cu -o $(BIN_DIR)/transformer

	mkdir -p $(LOG_DIR)/transformer

	nvidia-smi | tee $(LOG_DIR)/transformer/nvidia.log

ifeq ($(DEBUG), 1)
	@echo ""
	./$(BIN_DIR)/transformer $(DATA_DIR)/transformer/config

else

	@echo ""
	./$(BIN_DIR)/transformer $(DATA_DIR)/transformer/config | tee \
		$(LOG_DIR)/transformer/basic_configuration.log

	@echo ""
	./$(BIN_DIR)/transformer $(DATA_DIR)/transformer/config_benchmark \
		benchmark=0 | tee $(LOG_DIR)/transformer/config_benchmark_tf32.log

	./$(BIN_DIR)/transformer $(DATA_DIR)/transformer/config_benchmark \
		benchmark=0 tf32=0 | tee \
		$(LOG_DIR)/transformer/config_benchmark_not_tf32.log

	@echo ""
	nsys profile --stats=true --delay 10 --force-overwrite=true -o \
		$(LOG_DIR)/transformer/nsys_profile_tf32 ./$(BIN_DIR)/transformer \
		$(DATA_DIR)/transformer/config_benchmark iterations=1000 | tee \
		$(LOG_DIR)/transformer/nsys_profile_tf32.log

	@echo ""
	nsys profile --stats=true --delay 10 --force-overwrite=true -o \
		$(LOG_DIR)/transformer/nsys_profile_not_tf32 ./$(BIN_DIR)/transformer \
		$(DATA_DIR)/transformer/config_benchmark iterations=1000 tf32=0 | tee \
		$(LOG_DIR)/transformer/nsys_profile_not_tf32.log

ifeq ($(NCU), 1)
	@echo ""
	ncu --kernel-name regex:"softmax|layernorm|dropout|adam|dense" \
		--target-processes all --set detailed --force-overwrite \
		--launch-skip-before-match 100 --launch-count 50 --kill=yes -o \
		$(LOG_DIR)/transformer/ncu_tf32 ./$(BIN_DIR)/transformer \
		$(DATA_DIR)/transformer/config_benchmark num_encoder_layers=1 \
		num_decoder_layers=1 | tee $(LOG_DIR)/transformer/ncu_tf32.log

	ncu --kernel-name regex:"softmax|layernorm|dropout|adam|dense" \
		--target-processes all --set detailed --force-overwrite \
		--launch-skip-before-match 100 --launch-count 50 --kill=yes -o \
		$(LOG_DIR)/transformer/ncu_not_tf32 ./$(BIN_DIR)/transformer \
		$(DATA_DIR)/transformer/config_benchmark num_encoder_layers=1 \
		num_decoder_layers=1 tf32=0| tee $(LOG_DIR)/transformer/ncu_not_tf32.log

endif
endif

# The `transformer_overview` target compiles and runs `main/transformer.cu` with
# different configurations. Each measurement in this target changes one of the
# parameters set in the config file. This target can be used to get an overview
# how the different parameters change the behavior of the transformer.
transformer_overview:
	mkdir -p $(DATA_DIR)
	python $(DATA_GENERATION_SCRIPTS_DIR)/99_transformer.py $(DATA_DIR) \
		$(TMPDIR)

	mkdir -p $(BIN_DIR)
	nvc++ $(CUDA_FLAGS) $(MAIN_DIR)/transformer.cu -o $(BIN_DIR)/transformer

	mkdir -p $(LOG_DIR)/transformer

	nvidia-smi | tee $(LOG_DIR)/transformer/overview.log

	echo "Default Configuration:" >> $(LOG_DIR)/transformer/overview.log
	./$(BIN_DIR)/transformer $(DATA_DIR)/transformer/config_benchmark 2>&1 | \
		 tee -a $(LOG_DIR)/transformer/overview.log

	echo -e "\ntf32=0:" >> $(LOG_DIR)/transformer/overview.log
	./$(BIN_DIR)/transformer $(DATA_DIR)/transformer/config_benchmark tf32=0 \
		2>&1 | tee -a $(LOG_DIR)/transformer/overview.log

	echo -e "\nbatchsize=50 max_batchsize=50:" >> \
		$(LOG_DIR)/transformer/overview.log
	./$(BIN_DIR)/transformer $(DATA_DIR)/transformer/config_benchmark \
		batchsize=50 max_batchsize=50 2>&1 | tee -a \
		$(LOG_DIR)/transformer/overview.log

	echo -e "\nsequence_length=7:" >> $(LOG_DIR)/transformer/overview.log
	./$(BIN_DIR)/transformer $(DATA_DIR)/transformer/config_benchmark \
		sequence_length=7 2>&1 | tee -a $(LOG_DIR)/transformer/overview.log

	echo -e "\nembedding_dim=256:" >> $(LOG_DIR)/transformer/overview.log
	./$(BIN_DIR)/transformer $(DATA_DIR)/transformer/config_benchmark \
		embedding_dim=256 2>&1 | tee -a $(LOG_DIR)/transformer/overview.log

	echo -e "\nnum_encoder_layers=12:" >> $(LOG_DIR)/transformer/overview.log
	./$(BIN_DIR)/transformer $(DATA_DIR)/transformer/config_benchmark \
		num_encoder_layers=12 2>&1 | tee -a $(LOG_DIR)/transformer/overview.log

	echo -e "\nnum_decoder_layers=3:" >> $(LOG_DIR)/transformer/overview.log
	./$(BIN_DIR)/transformer $(DATA_DIR)/transformer/config_benchmark \
		num_decoder_layers=3 2>&1 | tee -a $(LOG_DIR)/transformer/overview.log

	echo -e "\nnum_heads=16:" >> $(LOG_DIR)/transformer/overview.log
	./$(BIN_DIR)/transformer $(DATA_DIR)/transformer/config_benchmark \
		num_heads=16 2>&1 | tee -a $(LOG_DIR)/transformer/overview.log

	echo -e "\nnum_heads=64:" >> $(LOG_DIR)/transformer/overview.log
	./$(BIN_DIR)/transformer $(DATA_DIR)/transformer/config_benchmark \
		num_heads=64 2>&1 | tee -a $(LOG_DIR)/transformer/overview.log

	echo -e "\nnum_embeddings=13760:" >> $(LOG_DIR)/transformer/overview.log
	./$(BIN_DIR)/transformer $(DATA_DIR)/transformer/config_benchmark \
		num_embeddings=13760 2>&1 | tee -a $(LOG_DIR)/transformer/overview.log

	echo -e "\nhidden_dim=4096:" >> $(LOG_DIR)/transformer/overview.log
	./$(BIN_DIR)/transformer $(DATA_DIR)/transformer/config_benchmark \
		hidden_dim=4096 2>&1 | tee -a $(LOG_DIR)/transformer/overview.log

# The `transformer_scaling` target compiles and runs `main/transformer.cu` with
# different configurations. Each measurement in this target varies one of the
# parameters set in the config file. This target can be used to get a series of
# measurements to see how the transformer behaves when changing a specific
# parameter.
#
# For more information see `SHELL_SCRIPTS_DIR/transformer_scaling.sh`.
transformer_scaling:
	mkdir -p $(DATA_DIR)
	python $(DATA_GENERATION_SCRIPTS_DIR)/99_transformer.py $(DATA_DIR) \
		$(TMPDIR)

	mkdir -p $(BIN_DIR)
	nvc++ $(CUDA_FLAGS) $(MAIN_DIR)/transformer.cu -o $(BIN_DIR)/transformer
	nvc++ $(CUDA_FLAGS) -DDEBUG $(MAIN_DIR)/transformer.cu -o \
		$(BIN_DIR)/transformer_test

	mkdir -p $(LOG_DIR)/transformer_scaling
	nvidia-smi | tee $(LOG_DIR)/transformer_scaling/nvidia.log

	./$(SHELL_SCRIPTS_DIR)/transformer_scaling.sh $(BIN_DIR) $(DATA_DIR) \
		$(LOG_DIR)

# The `weights_update` target compiles, runs `main/weights_update.cu`, and
# benchmarks the code.
#   - If `DEBUG` is set to 1, only one single configuration is run.
#     Additionally, the code will be compiled with the `-DDEBUG` flag.
#   - If `NCU` is set to 0, the Nvidia Compute Utility kernel measurement will
#     be omitted.
#
# Usage:
#    `make weights_update [DEBUG=0|1] [NCU=0|1]`
weights_update:
	mkdir -p $(BIN_DIR)
	nvc++ $(CUDA_FLAGS) $(MAIN_DIR)/weights_update.cu -o \
		$(BIN_DIR)/weights_update

	mkdir -p $(LOG_DIR)/weights_update

	nvidia-smi | tee $(LOG_DIR)/weights_update/nvidia.log

ifeq ($(DEBUG), 1)
	@echo ""
	./$(BIN_DIR)/weights_update N=99999999 iterations=3 warmup_steps=0

else
	@echo ""
	./$(BIN_DIR)/weights_update N=99999999 iterations=100 warmup_steps=10 | \
		tee $(LOG_DIR)/weights_update/weights_update.log

	@echo ""
	nsys profile --stats=true --force-overwrite=true -o \
		$(LOG_DIR)/weights_update/nsys_profile ./$(BIN_DIR)/weights_update \
		N=99999999 iterations=100 warmup_steps=0 | tee \
		$(LOG_DIR)/weights_update/nsys_profile.log

ifeq ($(NCU), 1)
	@echo ""
	ncu --target-processes all --set detailed --force-overwrite --kernel-name \
		regex:"^(?!_).*(?:[kK]ernel|[uU]pdate).*" -o \
		$(LOG_DIR)/weights_update/ncu ./$(BIN_DIR)/weights_update N=99999999 \
		iterations=5 warmup_steps=0 | tee $(LOG_DIR)/weights_update/ncu.log

endif
endif

# The `dropout` target compiles, runs `main/dropout.cu`, and benchmarks the
# code.
#   - If `DEBUG` is set to 1, only one single configuration is run.
#     Additionally, the code will be compiled with the `-DDEBUG` flag.
#   - If `NCU` is set to 0, the Nvidia Compute Utility kernel measurement will
#     be omitted.
#
# Usage:
#    `make dropout [DEBUG=0|1] [NCU=0|1]`
dropout:
	mkdir -p $(BIN_DIR)
	nvc++ $(CUDA_FLAGS) $(MAIN_DIR)/dropout.cu -o $(BIN_DIR)/dropout

	mkdir -p $(LOG_DIR)/dropout

	nvidia-smi | tee $(LOG_DIR)/dropout/nvidia.log

ifeq ($(DEBUG), 1)
	@echo ""
	./$(BIN_DIR)/dropout N=99999999 iterations=3 warmup_steps=0
else
	@echo ""
	./$(BIN_DIR)/dropout N=99999999 iterations=100 warmup_steps=10 | tee \
		$(LOG_DIR)/dropout/dropout.log

	@echo ""
	nsys profile --stats=true --force-overwrite=true -o \
		$(LOG_DIR)/dropout/nsys_profile ./$(BIN_DIR)/dropout N=99999999 \
		iterations=100 warmup_steps=0 | tee $(LOG_DIR)/dropout/nsys_profile.log

ifeq ($(NCU), 1)
	@echo ""
	ncu --target-processes all --set detailed --force-overwrite -o \
		$(LOG_DIR)/dropout/ncu ./$(BIN_DIR)/dropout N=99999999 iterations=5 \
		warmup_steps=0 | tee $(LOG_DIR)/dropout/ncu.log

endif
endif

# The `matrix_multiplication` target compiles, runs
# `main/matrix_multiplication.cu`, and benchmarks the code.
#   - If `DEBUG` is set to 1, only one single configuration is run.
#     Additionally, the code will be compiled with the `-DDEBUG` flag.
#   - If `NCU` is set to 0, the Nvidia Compute Utility kernel measurement will
#     be omitted.
#
# Usage:
#    `make matrix_multiplication [DEBUG=0|1] [NCU=0|1]`
matrix_multiplication:
	mkdir -p $(BIN_DIR)
	nvc++ $(CUDA_FLAGS) $(MAIN_DIR)/matrix_multiplication.cu -o \
		$(BIN_DIR)/matrix_multiplication

	mkdir -p $(LOG_DIR)/matrix_multiplication

	nvidia-smi | tee $(LOG_DIR)/matrix_multiplication/nvidia.log

ifeq ($(DEBUG), 1)
	@echo ""
	./$(BIN_DIR)/matrix_multiplication L=5000 M=50000 N=5000 iterations=3 \
		warmup_steps=0

else
	@echo ""
	./$(BIN_DIR)/matrix_multiplication L=5000 M=50000 N=5000 iterations=100 \
		warmup_steps=10 | tee \
		$(LOG_DIR)/matrix_multiplication/matrix_multiplication_1.log

	@echo ""
	./$(BIN_DIR)/matrix_multiplication L=5000 M=5000 N=5000 iterations=100 \
		warmup_steps=10 | tee \
		$(LOG_DIR)/matrix_multiplication/matrix_multiplication_2.log

	@echo ""
	./$(BIN_DIR)/matrix_multiplication L=5000 M=500 N=5000 iterations=100 \
		warmup_steps=10 | tee \
		$(LOG_DIR)/matrix_multiplication/matrix_multiplication_3.log

ifeq ($(NCU), 1)
	@echo ""
	ncu --target-processes all --set detailed --force-overwrite -o \
		$(LOG_DIR)/matrix_multiplication/ncu_1 \
		./$(BIN_DIR)/matrix_multiplication L=5000 M=50000 N=5000 iterations=5 \
		warmup_steps=0 | tee $(LOG_DIR)/matrix_multiplication/ncu_1.log

	@echo ""
	ncu --target-processes all --set detailed --force-overwrite -o \
		$(LOG_DIR)/matrix_multiplication/ncu_2 \
		./$(BIN_DIR)/matrix_multiplication L=5000 M=5000 N=5000 iterations=5 \
		warmup_steps=0 | tee $(LOG_DIR)/matrix_multiplication/ncu_2.log

	@echo ""
	ncu --target-processes all --set detailed --force-overwrite -o \
		$(LOG_DIR)/matrix_multiplication/ncu_3 \
		./$(BIN_DIR)/matrix_multiplication L=5000 M=500 N=5000 iterations=5 \
		warmup_steps=0 | tee $(LOG_DIR)/matrix_multiplication/ncu_3.log

endif
endif

# The `softmax` target compiles, runs `main/softmax.cu`, and benchmarks the
# code.
#   - If `DEBUG` is set to 1, only one single configuration is run.
#     Additionally, the code will be compiled with the `-DDEBUG` flag.
#   - If `NCU` is set to 0, the Nvidia Compute Utility kernel measurement will
#     be omitted.
#
# Usage:
#    `make softmax [DEBUG=0|1] [NCU=0|1]`
softmax:
	mkdir -p $(BIN_DIR)
	nvc++ $(CUDA_FLAGS) $(MAIN_DIR)/softmax.cu -o $(BIN_DIR)/softmax

	mkdir -p $(LOG_DIR)/softmax

	nvidia-smi | tee $(LOG_DIR)/softmax/nvidia.log

ifeq ($(DEBUG), 1)
	@echo ""
	./$(BIN_DIR)/softmax M=10000 N=100000 iterations=3 warmup_steps=0

else
	@echo ""
	./$(BIN_DIR)/softmax M=10000 N=100000 iterations=100 warmup_steps=10 | tee \
		$(LOG_DIR)/softmax/softmax_1.log

	@echo ""
	./$(BIN_DIR)/softmax M=10000 N=10000 iterations=100 warmup_steps=10 | tee \
		$(LOG_DIR)/softmax/softmax_2.log

	@echo ""
	./$(BIN_DIR)/softmax M=100000 N=10000 iterations=100 warmup_steps=10 | tee \
		$(LOG_DIR)/softmax/softmax_3.log

ifeq ($(NCU), 1)
	@echo ""
	ncu --target-processes all --set detailed --force-overwrite -o \
		$(LOG_DIR)/softmax/ncu_1 ./$(BIN_DIR)/softmax M=10000 N=100000 \
		iterations=5 warmup_steps=0 | tee $(LOG_DIR)/softmax/ncu_1.log

	@echo ""
	ncu --target-processes all --set detailed --force-overwrite -o \
		$(LOG_DIR)/softmax/ncu_2 ./$(BIN_DIR)/softmax M=10000 N=10000 \
		iterations=5 warmup_steps=0 | tee $(LOG_DIR)/softmax/ncu_2.log

	@echo ""
	ncu --target-processes all --set detailed --force-overwrite -o \
		$(LOG_DIR)/softmax/ncu_3 ./$(BIN_DIR)/softmax M=100000 N=10000 \
		iterations=5 warmup_steps=0 | tee $(LOG_DIR)/softmax/ncu_3.log

endif
endif

# The `atomic` target compiles and runs `main/atomic.cu`.
atomic:
	mkdir -p $(BIN_DIR)
	nvc++ $(CUDA_FLAGS) $(MAIN_DIR)/atomic.cu -o $(BIN_DIR)/atomic
	./$(BIN_DIR)/atomic

# The `stream` target compiles and runs `main/stream.cu`.
stream:
	mkdir -p $(BIN_DIR)
	nvc++ $(CUDA_FLAGS) $(MAIN_DIR)/stream.cu -o $(BIN_DIR)/stream

	mkdir -p $(LOG_DIR)/stream

	nsys profile --stats=true --force-overwrite=true -o \
		$(LOG_DIR)/stream/nsys_profile ./$(BIN_DIR)/stream | tee \
		$(LOG_DIR)/stream/nsys_profile.log

# The `mnist` target compiles, runs `main/mnist.cu`, and benchmarks the code.
#   - If `DEBUG` is set to 1, only one single configuration is run.
#     Additionally, the code will be compiled with the `-DDEBUG` flag.
#
# Usage:
#    `make mnist [DEBUG=0|1]`
mnist:
	mkdir -p $(DATA_DIR)
	python $(DATA_GENERATION_SCRIPTS_DIR)/98_mnist.py $(DATA_DIR) $(TMPDIR)

	mkdir -p $(BIN_DIR)
	nvc++ $(CUDA_FLAGS) $(MAIN_DIR)/mnist.cu -o $(BIN_DIR)/mnist

	mkdir -p $(LOG_DIR)/mnist

	nvidia-smi | tee $(LOG_DIR)/mnist/nvidia.log

ifeq ($(DEBUG), 1)

	@echo ""
	./$(BIN_DIR)/mnist $(DATA_DIR)/mnist/config

else

	@echo ""
	./$(BIN_DIR)/mnist $(DATA_DIR)/mnist/config | tee \
		$(LOG_DIR)/mnist/basic_configuration.log

	@echo ""
	./$(BIN_DIR)/mnist $(DATA_DIR)/mnist/config_benchmark benchmark=0 | tee \
		$(LOG_DIR)/mnist/config_benchmark_tf32.log

	./$(BIN_DIR)/mnist $(DATA_DIR)/mnist/config_benchmark benchmark=0 tf32=0 | \
		tee $(LOG_DIR)/mnist/config_benchmark_not_tf32.log

	@echo ""
	nsys profile --stats=true --delay 10 --force-overwrite=true -o \
		$(LOG_DIR)/mnist/nsys_profile_tf32 ./$(BIN_DIR)/mnist \
		$(DATA_DIR)/mnist/config_benchmark | tee \
		$(LOG_DIR)/mnist/nsys_profile_tf32.log

	@echo ""
	nsys profile --stats=true --delay 10 --force-overwrite=true -o \
		$(LOG_DIR)/mnist/nsys_profile_not_tf32 ./$(BIN_DIR)/mnist \
		$(DATA_DIR)/mnist/config_benchmark tf32=0 | tee \
		$(LOG_DIR)/mnist/nsys_profile_not_tf32.log

endif

# The `mnist_extended` target compiles, runs `main/mnist_extended.cu`, and
# benchmarks the code.
#   - If `DEBUG` is set to 1, only one single configuration is run.
#     Additionally, the code will be compiled with the `-DDEBUG` flag.
#
# Usage:
#    `make mnist_extended [DEBUG=0|1]`
mnist_extended:
	mkdir -p $(DATA_DIR)
	python $(DATA_GENERATION_SCRIPTS_DIR)/98_mnist.py $(DATA_DIR) $(TMPDIR)

	mkdir -p $(BIN_DIR)
	nvc++ $(CUDA_FLAGS) $(MAIN_DIR)/mnist_extended.cu -o \
		$(BIN_DIR)/mnist_extended

	mkdir -p $(LOG_DIR)/mnist_extended

	nvidia-smi | tee $(LOG_DIR)/mnist_extended/nvidia.log

ifeq ($(DEBUG), 1)

	@echo ""
	./$(BIN_DIR)/mnist_extended $(DATA_DIR)/mnist/config

else

	@echo ""
	./$(BIN_DIR)/mnist_extended $(DATA_DIR)/mnist/config | tee \
		$(LOG_DIR)/mnist_extended/basic_configuration.log

	@echo ""
	./$(BIN_DIR)/mnist_extended $(DATA_DIR)/mnist/config_benchmark benchmark=0 \
		| tee $(LOG_DIR)/mnist_extended/config_benchmark_tf32.log

	./$(BIN_DIR)/mnist_extended $(DATA_DIR)/mnist/config_benchmark benchmark=0 \
		tf32=0 | tee $(LOG_DIR)/mnist_extended/config_benchmark_not_tf32.log

	@echo ""
	nsys profile --stats=true --delay 10 --force-overwrite=true -o \
		$(LOG_DIR)/mnist_extended/nsys_profile_tf32 \
		./$(BIN_DIR)/mnist_extended $(DATA_DIR)/mnist/config_benchmark | tee \
		$(LOG_DIR)/mnist_extended/nsys_profile_tf32.log

	@echo ""
	nsys profile --stats=true --delay 10 --force-overwrite=true -o \
		$(LOG_DIR)/mnist_extended/nsys_profile_not_tf32 \
		./$(BIN_DIR)/mnist_extended $(DATA_DIR)/mnist/config_benchmark tf32=0 \
		| tee $(LOG_DIR)/mnist_extended/nsys_profile_not_tf32.log

endif

# The `plot` target creates plots from the in the `transformer_scaling` target
# created log files.
plot:
	python $(MAIN_DIR)/plot.py $(LOG_DIR)

# The `clean` target cleans the `BIN_DIR` and `DATA_DIR` directory.
clean:
	rm -r $(BIN_DIR)
	rm -r $(DATA_DIR)
