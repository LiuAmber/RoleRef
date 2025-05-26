#!/bin/bash

# Simple runner for activation extractor

# Check dependencies
python3 -c "import torch, transformers, tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Missing required packages. Install with:"
    echo "pip install torch transformers tqdm"
    exit 1
fi

# Default configurations (from your original code)
DEFAULT_QWEN_MODEL="Qwen2.5-7B-Instruct"
DEFAULT_DATA="../../data/test.json"
DEFAULT_SAVE="../../data/repe-qwen7b"

# Usage function
show_usage() {
    echo "Usage: $0 [model_type] [model_path] [data_path] [save_path]"
    echo ""
    echo "Examples:"
    echo "  $0 ft                           # Use default fine-tuned model"
    echo "  $0 repe                         # Use default with REPE"
    echo "  $0 ft /path/to/model            # Custom model path"
    echo "  $0 lora /path/to/lora data.json # Custom LoRA with data"
}

# Set parameters
MODEL_TYPE=${1:-"ft"}
MODEL_PATH=${2:-$DEFAULT_QWEN_MODEL}
DATA_PATH=${3:-$DEFAULT_DATA}
SAVE_PATH=${4:-"$DEFAULT_SAVE/${MODEL_TYPE}_activations"}

echo "=== Activation Extraction ==="
echo "Model type: $MODEL_TYPE"
echo "Model path: $MODEL_PATH"
echo "Data path: $DATA_PATH"
echo "Save path: $SAVE_PATH"

# Validate inputs
if [[ ! "$MODEL_TYPE" =~ ^(ft|lora|repe)$ ]]; then
    echo "Error: Model type must be ft, lora, or repe"
    show_usage
    exit 1
fi

if [[ ! -f "$DATA_PATH" ]]; then
    echo "Error: Data file not found: $DATA_PATH"
    exit 1
fi

# Create save directory
mkdir -p "$SAVE_PATH"

# Run extraction
echo "Starting activation extraction..."
python3 collect_activation.py \
    -m "$MODEL_PATH" \
    -t "$MODEL_TYPE" \
    -s "$SAVE_PATH" \
    -d "$DATA_PATH" \
    --verbose

if [ $? -eq 0 ]; then
    echo "✓ Activation extraction completed!"
    echo "Results saved to: $SAVE_PATH"
else
    echo "✗ Extraction failed"
    exit 1
fi