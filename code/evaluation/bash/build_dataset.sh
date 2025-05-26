#!/bin/bash

# Simple runner for evaluation dataset builder

# Check if Python and required packages are available
python3 -c "import json, glob, argparse, tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Missing required packages. Please install:"
    echo "pip install tqdm"
    exit 1
fi

# Set default parameters (from your original code examples)
DEFAULT_MODELS=(
    "llama-7b-repe:../../data/generate/representation/llama_test/*.json"
    "qwen-7b-repe:../../data/generate/representation/qwen_test/*.json"
    "qwen-32b-prompt:../../data/generate/prompt/qwen_32/*.json"
)

DEFAULT_OUTPUT_DIR="../../data/evaluation"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -i <input_path>     Input JSON file pattern"
    echo "  -o <output_file>    Output JSONL file"
    echo "  -m <model_name>     Model name"
    echo "  --all              Build all default models"
    echo "  -v                 Verbose output"
    echo "  -h                 Show this help"
    echo ""
    echo "Examples:"
    echo "  # Build single model dataset"
    echo "  $0 -i '/path/to/*.json' -o output.jsonl -m 'model_name'"
    echo ""
    echo "  # Build all default models"
    echo "  $0 --all"
    echo ""
    echo "  # Custom usage"
    echo "  $0 -i input.json -o dataset.jsonl -m llama-7b -v"
}

# Function to build single dataset
build_single() {
    local input_path="$1"
    local output_file="$2"
    local model_name="$3"
    local verbose="$4"
    
    echo "Building dataset for model: $model_name"
    echo "Input: $input_path"
    echo "Output: $output_file"
    
    # Create output directory
    mkdir -p "$(dirname "$output_file")"
    
    # Build command
    local cmd="python3 construct_evaluation_data.py -i \"$input_path\" -o \"$output_file\" -m \"$model_name\""
    if [ "$verbose" = "true" ]; then
        cmd="$cmd -v"
    fi
    
    # Execute
    eval $cmd
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully built dataset for $model_name"
    else
        echo "✗ Failed to build dataset for $model_name"
        return 1
    fi
}

# Function to build all default models
build_all() {
    local verbose="$1"
    
    echo "Building datasets for all default models..."
    
    for model_info in "${DEFAULT_MODELS[@]}"; do
        IFS=':' read -r model_name input_path <<< "$model_info"
        output_file="$DEFAULT_OUTPUT_DIR/${model_name}_dataset.jsonl"
        
        echo ""
        echo "=== Processing $model_name ==="
        
        if build_single "$input_path" "$output_file" "$model_name" "$verbose"; then
            echo "✓ Completed $model_name"
        else
            echo "✗ Failed $model_name"
        fi
    done
    
    echo ""
    echo "✓ All models processed!"
}

# Parse command line arguments
INPUT_PATH=""
OUTPUT_FILE=""
MODEL_NAME=""
VERBOSE="false"
BUILD_ALL="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        -i)
            INPUT_PATH="$2"
            shift 2
            ;;
        -o)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -m)
            MODEL_NAME="$2"
            shift 2
            ;;
        --all)
            BUILD_ALL="true"
            shift
            ;;
        -v)
            VERBOSE="true"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Execute based on options
if [ "$BUILD_ALL" = "true" ]; then
    build_all "$VERBOSE"
elif [ -n "$INPUT_PATH" ] && [ -n "$OUTPUT_FILE" ] && [ -n "$MODEL_NAME" ]; then
    build_single "$INPUT_PATH" "$OUTPUT_FILE" "$MODEL_NAME" "$VERBOSE"
else
    echo "Error: Missing required parameters"
    echo ""
    show_usage
    exit 1
fi