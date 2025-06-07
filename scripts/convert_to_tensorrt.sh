#!/bin/bash

# Script to convert ONNX model to TensorRT engine
# Usage: ./scripts/convert_to_tensorrt.sh input.onnx output.trt [batch_size] [precision]

set -e  # Exit on any error

# Default values
DEFAULT_BATCH_SIZE=1
DEFAULT_PRECISION="fp16"
DEFAULT_WORKSPACE_SIZE=1073741824  # 1GB in bytes

# Parse arguments
ONNX_PATH=$1
TRT_PATH=$2
BATCH_SIZE=${3:-$DEFAULT_BATCH_SIZE}
PRECISION=${4:-$DEFAULT_PRECISION}

# Validate arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <onnx_path> <trt_path> [batch_size] [precision]"
    echo "  onnx_path:    Path to input ONNX model"
    echo "  trt_path:     Path to output TensorRT engine"
    echo "  batch_size:   Batch size for optimization (default: $DEFAULT_BATCH_SIZE)"
    echo "  precision:    Precision mode: fp32, fp16, int8 (default: $DEFAULT_PRECISION)"
    echo ""
    echo "Example:"
    echo "  $0 models/flower_classifier.onnx models/flower_classifier.trt 4 fp16"
    exit 1
fi

# Check if input file exists
if [ ! -f "$ONNX_PATH" ]; then
    echo "Error: ONNX file '$ONNX_PATH' not found!"
    exit 1
fi

# Check if trtexec is available
if ! command -v trtexec &> /dev/null; then
    echo "Error: trtexec not found!"
    echo "Please install TensorRT and ensure trtexec is in your PATH"
    echo "Installation guide: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$TRT_PATH")
mkdir -p "$OUTPUT_DIR"

echo "Converting ONNX to TensorRT..."
echo "Input:  $ONNX_PATH"
echo "Output: $TRT_PATH"
echo "Batch:  $BATCH_SIZE"
echo "Precision: $PRECISION"
echo ""

# Build precision flags
PRECISION_FLAGS=""
case $PRECISION in
    "fp16")
        PRECISION_FLAGS="--fp16"
        ;;
    "int8")
        PRECISION_FLAGS="--int8"
        echo "Warning: INT8 precision requires calibration dataset for optimal results"
        ;;
    "fp32")
        # Default precision, no flags needed
        ;;
    *)
        echo "rror: Invalid precision '$PRECISION'. Use: fp32, fp16, or int8"
        exit 1
        ;;
esac

# Run trtexec conversion
echo "Running TensorRT conversion..."
trtexec \
    --onnx="$ONNX_PATH" \
    --saveEngine="$TRT_PATH" \
    --shapes=input:${BATCH_SIZE}x3x224x224 \
    --workspace=$DEFAULT_WORKSPACE_SIZE \
    $PRECISION_FLAGS \
    --verbose

# Check if conversion was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "TensorRT conversion completed successfully!"
    echo "Engine saved to: $TRT_PATH"

    # Display file info
    if [ -f "$TRT_PATH" ]; then
        FILE_SIZE=$(du -h "$TRT_PATH" | cut -f1)
        echo "Engine size: $FILE_SIZE"
    fi

    echo ""
    echo "To test the engine, you can use:"
else
    echo ""
    echo "TensorRT conversion failed!"
    exit 1
fi
