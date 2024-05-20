#!/bin/bash

# Check if all necessary directories are provided as arguments
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <train_tar_directory> <validation_tar_directory> <train_output_directory> <validation_output_directory>"
    exit 1
fi

# Set the directories from the arguments
TRAIN_TAR_DIR=$1
VALIDATION_TAR_DIR=$2
TRAIN_OUTPUT_DIR=$3
VALIDATION_OUTPUT_DIR=$4

# Create output directories if they don't exist
mkdir -p $TRAIN_OUTPUT_DIR
mkdir -p $VALIDATION_OUTPUT_DIR

# Function to extract .tar files from a directory into a specified output directory
extract_tars() {
    local TAR_DIR=$1
    local OUTPUT_DIR=$2
    
    for tar_file in "$TAR_DIR"/*.tar; do
        echo "Extracting $tar_file to $OUTPUT_DIR"
        tar -xvf "$tar_file" -C "$OUTPUT_DIR"
    done
}

# Extract train .tar files
extract_tars $TRAIN_TAR_DIR $TRAIN_OUTPUT_DIR

# Extract validation .tar files
extract_tars $VALIDATION_TAR_DIR $VALIDATION_OUTPUT_DIR

echo "Extraction complete. Train data is in $TRAIN_OUTPUT_DIR, and validation data is in $VALIDATION_OUTPUT_DIR."
