#!/bin/bash

# Check if train and validation directories are provided as arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <train_download_directory> <validation_download_directory>"
    exit 1
fi

# Set the download directories from the arguments
TRAIN_DOWNLOAD_DIR=$1
VALIDATION_DOWNLOAD_DIR=$2

# Base URLs
TRAIN_BASE_URL="https://huggingface.co/datasets/pixparse/cc3m-wds/resolve/main"
VALIDATION_BASE_URL="https://huggingface.co/datasets/pixparse/cc3m-wds/resolve/main"

# Create download directories if they don't exist
mkdir -p $TRAIN_DOWNLOAD_DIR
mkdir -p $VALIDATION_DOWNLOAD_DIR

# Loop to download train files
for i in $(seq -f "%04g" 0 575); do
    FILE_URL="${TRAIN_BASE_URL}/cc3m-train-${i}.tar"
    wget -P $TRAIN_DOWNLOAD_DIR $FILE_URL
done

# Loop to download validation files
for i in $(seq -f "%04g" 0 15); do
    FILE_URL="${VALIDATION_BASE_URL}/cc3m-validation-${i}.tar"
    wget -P $VALIDATION_DOWNLOAD_DIR $FILE_URL
done

echo "All .tar files have been downloaded to $TRAIN_DOWNLOAD_DIR and $VALIDATION_DOWNLOAD_DIR."
