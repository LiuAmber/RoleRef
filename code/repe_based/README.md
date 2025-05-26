# Representation Analysis and Control Toolkit

## Overview

This toolkit provides comprehensive tools for analyzing and controlling neural model representations in roleplay experiments. The toolkit enables extraction of model activations, training of classification models on these representations, and generation of controlled responses through representation editing techniques.

## Components

### 1. Activation Collection (`collect_activation.py`)
Extracts hidden state activations from language models across different question types to analyze internal representations and compute representation differences.

### 2. Representation Classifier (`classify.py`) 
Trains neural network classifiers on extracted activations to analyze the separability of different response types and identify optimal layers for representation intervention.

### 3. Controlled Response Generator (`generate_with_representation_control.py`)
Generates model responses with representation editing applied, enabling controlled modification of model behavior through activation manipulation.

### 4. Representation Control Framework (`rep_control_reading_vec.py`)
Provides the underlying framework for wrapping models and implementing representation editing through activation injection and manipulation.

## Installation Requirements

### Core Dependencies
```bash
pip install torch transformers pandas openpyxl tqdm numpy scikit-learn pickle
```

### GPU Requirements
These tools require CUDA-compatible GPUs for efficient processing. Memory requirements vary by model size, with larger models requiring proportionally more GPU memory.

## Tool Descriptions

### Activation Collection

**Purpose**: Extracts model activations for analysis and representation editing preparation.

**Key Features**:
- Supports multiple model types including fine-tuned, LoRA, and representation editing models
- Processes five question categories: answerable, out_series, fake, absent, context_conflict
- Computes representation differences between answerable and problematic questions
- Generates common representation patterns for intervention

**Usage**:
```bash
python3 collect_activation.py -m /path/to/model -t ft -s ./activations -d data.json
```

**Output**: Pickle files containing layer-wise activations and computed representation differences for each character role.

### Representation Classifier

**Purpose**: Analyzes the separability of different question types using extracted activations.

**Key Features**:
- Implements simple fully-connected neural networks for binary classification
- Evaluates classification performance across all model layers
- Supports multiple question type combinations for analysis
- Generates comprehensive performance reports per layer and character

**Usage**:
```bash
python3 classify.py --data_path /path/to/activations --seed 42
```

**Output**: Excel files containing classification accuracies across layers, roles, and question types.

### Controlled Response Generator

**Purpose**: Generates model responses with applied representation editing to modify behavior.

**Key Features**:
- Implements representation editing through activation manipulation
- Supports configurable intervention coefficients and layer ranges
- Processes complete datasets with controlled generation
- Provides both direct and signal-based query modes

**Usage**:
```bash
python3 generate_with_representation_control.py \
    --model_path /path/to/model \
    --data_path /path/to/questions.json \
    --save_path ./controlled_outputs \
    --ceoff -0.1 \
    --safe_pattern_path /path/to/patterns/{role}/rep_diff_0.3.pkl
```

**Output**: JSON files containing generated responses for each character and question type with applied representation control.

### Representation Control Framework

**Purpose**: Provides the underlying infrastructure for model wrapping and activation manipulation.

**Key Features**:
- Wraps language models to enable activation intervention
- Supports multiple intervention operators including linear combination and piecewise linear
- Handles token position specification and masking
- Provides activation normalization and extraction capabilities

**Integration**: This framework is automatically utilized by other tools and does not require direct invocation.

## Workflow Integration

### Standard Analysis Pipeline

**Step 1: Extract Activations**
```bash
python3 collect_activation.py -m /path/to/model -t ft -s ./activations -d questions.json
```

**Step 2: Train Classifiers**
```bash
python3 classify.py --data_path ./activations --seed 42
```

**Step 3: Generate Controlled Responses**
```bash
python3 generate_with_representation_control.py \
    --model_path /path/to/model \
    --data_path questions.json \
    --save_path ./controlled_responses \
    --safe_pattern_path ./activations/{role}/rep_diff_0.3.pkl
```

### Configuration Parameters

**Activation Extraction**:
- `model_type`: Specifies model architecture (ft, lora, repe)
- `max_samples`: Controls number of samples processed per role
- Model-specific hidden dimensions are automatically detected

**Classification Analysis**:
- `seed`: Ensures reproducible results across runs
- Layer analysis spans full model depth automatically
- Multiple question type combinations are evaluated systematically

**Controlled Generation**:
- `ceoff`: Controls intervention strength (typical range: -0.3 to 0.3)
- `safe_pattern_path`: Specifies representation patterns for intervention
- `query_type`: Determines prompt formatting approach

## Data Format Requirements

### Input Question Format
```json
{
  "character_name": {
    "answerable": [
      {"question": "...", "fake_method": ""}
    ],
    "fake": [
      {"question": "...", "fake_method": "concept_replacement"}
    ]
  }
}
```

### Activation Output Format
Each character directory contains:
- `rep_answerable.pkl`: Activations for answerable questions
- `rep_fake.pkl`: Activations for fake questions  
- `rep_diff.pkl`: Raw representation differences
- `rep_diff_0.3.pkl`: Processed patterns for intervention

### Classification Results Format
Excel files with columns for role, layer, training accuracy, validation accuracy, and test accuracies across question types.

## Performance Considerations

### Memory Management
Large language models require substantial GPU memory. The toolkit implements automatic memory clearing and supports batch processing to manage resource constraints.

### Processing Time
Activation extraction scales with model size and sample count. Classification training is relatively fast. Controlled generation time depends on response length and intervention complexity.

### Reproducibility
All tools support random seed specification for consistent results across experimental runs.

## Model Compatibility

### Supported Architectures
- Qwen series models (2.5-7B, 2-72B variants)
- Llama series models (3-8B, 3.1-8B variants)  
- Mistral models (7B variants)

### Adaptation Requirements
Model-specific configurations are automatically detected based on model paths and architecture specifications. Hidden dimension sizes are extracted programmatically to ensure compatibility.

## Troubleshooting

### Common Issues

**CUDA Memory Errors**: Reduce batch sizes or sample counts. Consider using gradient checkpointing for large models.

**Import Errors**: Ensure all dependency modules are available in the Python path. Verify transformers library compatibility with model architectures.

**File Path Issues**: Use absolute paths for model and data locations. Ensure all required directories exist before execution.

**Classification Performance**: Low accuracies may indicate insufficient representation differences or inappropriate layer selection. Consider adjusting intervention patterns or model architectures.

### Performance Optimization

For optimal performance, ensure adequate GPU memory allocation, use appropriate batch sizes for hardware constraints, and leverage multiple GPUs when available through device mapping configurations.

## Output Analysis

Classification results indicate which model layers contain the most separable representations for different question types. Higher accuracies suggest stronger representation differences that can be leveraged for effective intervention.

Controlled generation outputs demonstrate the impact of representation editing on model behavior. Successful interventions should show modified responses that align with desired behavioral changes while maintaining response quality and character consistency.