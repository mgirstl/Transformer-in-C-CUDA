#!/bin/bash

# This script runs scaling tests for various parameters of the transformer
# model. It loops through different values for batchsize, sequence_length,
# embedding_dim, num_encoder_layers, num_decoder_layers, num_heads and
# num_embeddings. For each parameter, it runs a test program and, if successful,
# logs the parameter value and runs a benchmark program.
#
# Usage:
#    `./run_scaling_tests.sh <BIN_DIR> <DATA_DIR> <LOG_DIR>`
#    - `BIN_DIR`: Directory containing the compiled executables.
#    - `DATA_DIR`: Directory containing the data to be used by the executables.
#    - `LOG_DIR`: Directory where log files will be stored.

# Arguments
BIN_DIR=$1
DATA_DIR=$2
LOG_DIR=$3

# Function to initialize log file
initialize_log_file() {
    local log_file=$1
    mkdir -p "$(dirname "$log_file")"
    > "$log_file" # Overwrite the log file if it already exists
}

# Loop through batch sizes
LOG_FILE="${LOG_DIR}/transformer_scaling/batchsize.log"
ERROR_LOG_FILE="${LOG_DIR}/transformer_scaling/batchsize_error.log"
initialize_log_file "$LOG_FILE"
initialize_log_file "$ERROR_LOG_FILE"
for ((batchsize = 10; batchsize <= 500; batchsize += 10)); do
    echo "Running batchsize: $batchsize"
    # Run the test program
    ./${BIN_DIR}/transformer_test ${DATA_DIR}/transformer/config_test \
        batchsize=$batchsize max_batchsize=$batchsize progress=0 \
        > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        # Log the batch size
        echo "batchsize: $batchsize" >> "$LOG_FILE"
        # Run the benchmark program
        ./${BIN_DIR}/transformer ${DATA_DIR}/transformer/config_benchmark \
            batchsize=$batchsize max_batchsize=$batchsize progress=0 \
            >> "$LOG_FILE" 2>&1
    else
        # Capture the error output and log it
        ERROR_OUTPUT=$(./${BIN_DIR}/transformer_test \
            ${DATA_DIR}/transformer/config_test batchsize=$batchsize \
            max_batchsize=$batchsize progress=0 2>&1)
        echo "batchsize: $batchsize" >> "$ERROR_LOG_FILE"
        echo "$ERROR_OUTPUT" >> "$ERROR_LOG_FILE"
        # Exit the loop if the test program fails
        break
    fi
done

# Loop through sequence lengths
LOG_FILE="${LOG_DIR}/transformer_scaling/sequence_length.log"
ERROR_LOG_FILE="${LOG_DIR}/transformer_scaling/sequence_length_error.log"
initialize_log_file "$LOG_FILE"
initialize_log_file "$ERROR_LOG_FILE"

for ((sequence_length = 1; sequence_length <= 100; sequence_length += 1)); do
    echo "Running sequence_length: $sequence_length"
    # Run the test program
    ./${BIN_DIR}/transformer_test ${DATA_DIR}/transformer/config_test \
        sequence_length=$sequence_length progress=0 > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        # Log the sequence length
        echo "sequence_length: $sequence_length" >> "$LOG_FILE"
        # Run the benchmark program
        ./${BIN_DIR}/transformer ${DATA_DIR}/transformer/config_benchmark \
            sequence_length=$sequence_length progress=0 >> "$LOG_FILE" 2>&1
    else
        # Capture the error output and log it
        ERROR_OUTPUT=$(./${BIN_DIR}/transformer_test \
            ${DATA_DIR}/transformer/config_test \
            sequence_length=$sequence_length progress=0 2>&1)
        echo "sequence_length: $sequence_length" >> "$ERROR_LOG_FILE"
        echo "$ERROR_OUTPUT" >> "$ERROR_LOG_FILE"
        # Exit the loop if the test program fails
        break
    fi
done

# Loop through embedding dimensions
LOG_FILE="${LOG_DIR}/transformer_scaling/embedding_dim.log"
ERROR_LOG_FILE="${LOG_DIR}/transformer_scaling/embedding_dim_error.log"
initialize_log_file "$LOG_FILE"
initialize_log_file "$ERROR_LOG_FILE"

for ((embedding_dim = 25; embedding_dim <= 10000; embedding_dim += 25)); do
    echo "Running embedding_dim: $embedding_dim"
    # Run the test program
    ./${BIN_DIR}/transformer_test ${DATA_DIR}/transformer/config_test \
        embedding_dim=$embedding_dim progress=0 > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        # Log the embedding dimension
        echo "embedding_dim: $embedding_dim" >> "$LOG_FILE"
        # Run the benchmark program
        ./${BIN_DIR}/transformer ${DATA_DIR}/transformer/config_benchmark \
            embedding_dim=$embedding_dim progress=0 >> "$LOG_FILE" 2>&1
    else
        # Capture the error output and log it
        ERROR_OUTPUT=$(./${BIN_DIR}/transformer_test \
            ${DATA_DIR}/transformer/config_test embedding_dim=$embedding_dim \
            progress=0 2>&1)
        echo "embedding_dim: $embedding_dim" >> "$ERROR_LOG_FILE"
        echo "$ERROR_OUTPUT" >> "$ERROR_LOG_FILE"
        # Exit the loop if the test program fails
        break
    fi
done

# Loop through hidden dimensions
LOG_FILE="${LOG_DIR}/transformer_scaling/hidden_dim.log"
ERROR_LOG_FILE="${LOG_DIR}/transformer_scaling/hidden_dim_error.log"
initialize_log_file "$LOG_FILE"
initialize_log_file "$ERROR_LOG_FILE"

for ((hidden_dim = 1000; hidden_dim <= 50000; hidden_dim += 1000)); do
    echo "Running hidden_dim: $hidden_dim"
    # Run the test program
    ./${BIN_DIR}/transformer_test ${DATA_DIR}/transformer/config_test \
        hidden_dim=$hidden_dim progress=0 > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        # Log the hidden dimension
        echo "hidden_dim: $hidden_dim" >> "$LOG_FILE"
        # Run the benchmark program
        ./${BIN_DIR}/transformer ${DATA_DIR}/transformer/config_benchmark \
            hidden_dim=$hidden_dim progress=0 >> "$LOG_FILE" 2>&1
    else
        # Capture the error output and log it
        ERROR_OUTPUT=$(./${BIN_DIR}/transformer_test \
            ${DATA_DIR}/transformer/config_test hidden_dim=$hidden_dim \
            progress=0 2>&1)
        echo "hidden_dim: $hidden_dim" >> "$ERROR_LOG_FILE"
        echo "$ERROR_OUTPUT" >> "$ERROR_LOG_FILE"
        # Exit the loop if the test program fails
        break
    fi
done

# Loop through number of encoder layers
LOG_FILE="${LOG_DIR}/transformer_scaling/num_encoder_layers.log"
ERROR_LOG_FILE="${LOG_DIR}/transformer_scaling/num_encoder_layers_error.log"
initialize_log_file "$LOG_FILE"
initialize_log_file "$ERROR_LOG_FILE"

for ((num_encoder_layers = 1; num_encoder_layers <= 50; num_encoder_layers += 1)); do
    echo "Running num_encoder_layers: $num_encoder_layers"
    # Run the test program
    ./${BIN_DIR}/transformer_test ${DATA_DIR}/transformer/config_test \
        num_encoder_layers=$num_encoder_layers progress=0 > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        # Log the number of encoder layers
        echo "num_encoder_layers: $num_encoder_layers" >> "$LOG_FILE"
        # Run the benchmark program
        ./${BIN_DIR}/transformer ${DATA_DIR}/transformer/config_benchmark \
            num_encoder_layers=$num_encoder_layers progress=0 \
            >> "$LOG_FILE" 2>&1
    else
        # Capture the error output and log it
        ERROR_OUTPUT=$(./${BIN_DIR}/transformer_test \
            ${DATA_DIR}/transformer/config_test \
            num_encoder_layers=$num_encoder_layers progress=0 2>&1)
        echo "num_encoder_layers: $num_encoder_layers" >> "$ERROR_LOG_FILE"
        echo "$ERROR_OUTPUT" >> "$ERROR_LOG_FILE"
        # Exit the loop if the test program fails
        break
    fi
done

# Loop through number of decoder layers
LOG_FILE="${LOG_DIR}/transformer_scaling/num_decoder_layers.log"
ERROR_LOG_FILE="${LOG_DIR}/transformer_scaling/num_decoder_layers_error.log"
initialize_log_file "$LOG_FILE"
initialize_log_file "$ERROR_LOG_FILE"

for ((num_decoder_layers = 1; num_decoder_layers <= 50; num_decoder_layers += 1)); do
    echo "Running num_decoder_layers: $num_decoder_layers"
    # Run the test program
    ./${BIN_DIR}/transformer_test ${DATA_DIR}/transformer/config_test \
        num_decoder_layers=$num_decoder_layers progress=0 > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        # Log the number of decoder layers
        echo "num_decoder_layers: $num_decoder_layers" >> "$LOG_FILE"
        # Run the benchmark program
        ./${BIN_DIR}/transformer ${DATA_DIR}/transformer/config_benchmark \
            num_decoder_layers=$num_decoder_layers progress=0 >> "$LOG_FILE" 2>&1
    else
        # Capture the error output and log it
        ERROR_OUTPUT=$(./${BIN_DIR}/transformer_test \
            ${DATA_DIR}/transformer/config_test \
            num_decoder_layers=$num_decoder_layers progress=0 2>&1)
        echo "num_decoder_layers: $num_decoder_layers" >> "$ERROR_LOG_FILE"
        echo "$ERROR_OUTPUT" >> "$ERROR_LOG_FILE"
        # Exit the loop if the test program fails
        break
    fi
done

# create num_heads log file
LOG_FILE="${LOG_DIR}/transformer_scaling/num_heads.log"
ERROR_LOG_FILE="${LOG_DIR}/transformer_scaling/num_heads_error.log"
initialize_log_file "$LOG_FILE"
initialize_log_file "$ERROR_LOG_FILE"

# Additional measurement for num_heads=8
num_heads=8
echo "Running num_heads: $num_heads"
./${BIN_DIR}/transformer_test ${DATA_DIR}/transformer/config_test \
    num_heads=$num_heads progress=0 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "num_heads: $num_heads" >> "$LOG_FILE"
    ./${BIN_DIR}/transformer ${DATA_DIR}/transformer/config_benchmark \
        num_heads=$num_heads progress=0 >> "$LOG_FILE" 2>&1
else
    # Capture the error output and log it
    ERROR_OUTPUT=$(./${BIN_DIR}/transformer_test \
        ${DATA_DIR}/transformer/config_test num_heads=$num_heads progress=0 \
        2>&1)
    echo "num_heads: $num_heads" >> "$ERROR_LOG_FILE"
    echo "$ERROR_OUTPUT" >> "$ERROR_LOG_FILE"
fi

# Loop through number of heads
for ((num_heads = 32; num_heads <= 512; num_heads += 32)); do
    echo "Running num_heads: $num_heads"
    ./${BIN_DIR}/transformer_test ${DATA_DIR}/transformer/config_test \
        num_heads=$num_heads progress=0 > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "num_heads: $num_heads" >> "$LOG_FILE"
        ./${BIN_DIR}/transformer ${DATA_DIR}/transformer/config_benchmark \
            num_heads=$num_heads progress=0 >> "$LOG_FILE" 2>&1
    else
        # Capture the error output and log it
        ERROR_OUTPUT=$(./${BIN_DIR}/transformer_test \
            ${DATA_DIR}/transformer/config_test num_heads=$num_heads \
            progress=0 2>&1)
        echo "num_heads: $num_heads" >> "$ERROR_LOG_FILE"
        echo "$ERROR_OUTPUT" >> "$ERROR_LOG_FILE"
        # Exit the loop if the test program fails
        break
    fi
done

# Create num_embeddings log file
LOG_FILE="${LOG_DIR}/transformer_scaling/num_embeddings.log"
ERROR_LOG_FILE="${LOG_DIR}/transformer_scaling/num_embeddings_error.log"
initialize_log_file "$LOG_FILE"
initialize_log_file "$ERROR_LOG_FILE"

# Additional measurement for num_embeddings=1000
num_embeddings=1000
echo "Running num_embeddings: $num_embeddings"
./${BIN_DIR}/transformer_test ${DATA_DIR}/transformer/config_test \
    num_embeddings=$num_embeddings progress=0 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "num_embeddings: $num_embeddings" >> "$LOG_FILE"
    ./${BIN_DIR}/transformer ${DATA_DIR}/transformer/config_benchmark \
        num_embeddings=$num_embeddings progress=0 >> "$LOG_FILE" 2>&1
else
    # Capture the error output and log it
    ERROR_OUTPUT=$(./${BIN_DIR}/transformer_test \
        ${DATA_DIR}/transformer/config_test num_embeddings=$num_embeddings \
        progress=0 2>&1)
    echo "num_embeddings: $num_embeddings" >> "$ERROR_LOG_FILE"
    echo "$ERROR_OUTPUT" >> "$ERROR_LOG_FILE"
fi

# Loop through the number of embeddings
for ((num_embeddings = 10000; num_embeddings <= 500000; num_embeddings += 10000)); do
    echo "Running num_embeddings: $num_embeddings"
    ./${BIN_DIR}/transformer_test ${DATA_DIR}/transformer/config_test \
        num_embeddings=$num_embeddings progress=0 > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "num_embeddings: $num_embeddings" >> "$LOG_FILE"
        ./${BIN_DIR}/transformer ${DATA_DIR}/transformer/config_benchmark \
            num_embeddings=$num_embeddings progress=0 >> "$LOG_FILE" 2>&1
    else
        # Capture the error output and log it
        ERROR_OUTPUT=$(./${BIN_DIR}/transformer_test \
            ${DATA_DIR}/transformer/config_test num_embeddings=$num_embeddings \
            progress=0 2>&1)
        echo "num_embeddings: $num_embeddings" >> "$ERROR_LOG_FILE"
        echo "$ERROR_OUTPUT" >> "$ERROR_LOG_FILE"
        break
    fi
done
