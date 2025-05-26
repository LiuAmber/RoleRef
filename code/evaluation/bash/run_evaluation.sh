#!/bin/bash

# Simple examples for roleplay evaluation tool

# Example 1: Basic usage with original paths
echo "=== Example 1: Basic Usage ==="
python3 get_evaluation_result.py \
    -i "../../data/evaluation/result/qwen32.json" \
    -o "../../data/evaluation/processed/qwen32" \
    --verbose

# Example 2: Process specific data types only
echo "=== Example 2: Specific Data Types ==="
python3 get_evaluation_result.py \
    -i "../../data/evaluation/result/qwen32.json" \
    -o "../../data/evaluation/processed/qwen32/output_test" \
    -t answerable fake \
    --verbose

# Example 3: Quick test with sample data
echo "=== Example 3: Sample Test ==="

# Create simple test data
cat > test_data.json << EOF
[
    {
        "role": "test_character",
        "data_type": "answerable", 
        "gpt_response": "Awareness of Pitfalls: 4.5 point\nRefusal to Answer Judgment: 3.2 point\nAlignment with Role Background: 4.0 point\nAlignment with Role Style: 3.8 point\nAlignment with Role Abilities: 4.2 point\nAlignment with Role Personality: 3.9 point\nConsistency of Response: 4.1 point\nQuality of Response: 3.7 point\nFactuality of Response: 4.3 point"
    }
]
EOF

python3 get_evaluation_result.py -i test_data.json -o ../../data/evaluation/processed/qwen32_repe/test_output --verbose

echo "✓ Test completed. Check ./test_output/ for results."