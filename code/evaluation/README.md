# Roleplay Evaluation Toolkit

A comprehensive toolkit for processing and evaluating roleplay experiment data.

## 🛠️ Tools

### 1. **Evaluation Dataset Builder** (`construct_evaluation_data.py`)
Constructs evaluation datasets from roleplay experiment results.

### 2. **Evaluation Results Processor** (`get_evaluation_result.py`)  
Processes GPT evaluation results and generates analysis reports.

## 📦 Quick Start

### Install Requirements
```bash
pip install pandas openpyxl tqdm
```

### Make Scripts Executable
```bash
chmod +x build_dataset.sh
chmod +x run_evaluation.sh
```

## 🔨 Tool 1: Dataset Builder

### Purpose
Converts roleplay experiment outputs into standardized evaluation prompts for GPT assessment.

### Basic Usage
```bash
# Build single dataset
python3 construct_evaluation_data.py -i "/path/to/data/*.json" -o output.jsonl -m "model_name"

# Build all default models
./build_dataset.sh --all

# Custom usage
./build_dataset.sh -i input.json -o dataset.jsonl -m llama-7b -v
```

### Input Format
```json
{
  "character_name": {
    "answerable": [
      {"question": "...", "response": "...", "fake_method": ""}
    ],
    "fake": [
      {"question": "...", "response": "...", "fake_method": "concept_replacement"}
    ]
  }
}
```

### Output Format (JSONL)
```json
{"question": "evaluation_prompt", "role": "character", "data_type": "answerable", "model": "llama-7b"}
```

## 📊 Tool 2: Results Processor  

### Purpose
Analyzes GPT evaluation responses and generates Excel reports with metrics.

### Basic Usage
```bash
# Process evaluation results
python3 get_evaluation_result.py -i results.json -o output_dir/

# Use default paths
./run_evaluation.sh

# Custom paths
./run_evaluation.sh /path/to/input.json /path/to/output/
```

### Input Format
```json
[
  {
    "role": "character_name",
    "data_type": "answerable", 
    "gpt_response": "Awareness of Pitfalls: 4.5 point\nRefusal to Answer Judgment: 3.2 point\n..."
  }
]
```

### Output Files
- `detail_result.xlsx` - Results by role
- `all_result.xlsx` - Aggregated results

## 📋 Command Reference

### Dataset Builder Options
| Option | Description | Example |
|--------|-------------|---------|
| `-i` | Input file pattern | `-i "*.json"` |
| `-o` | Output JSONL file | `-o dataset.jsonl` |
| `-m` | Model name | `-m "llama-7b"` |
| `--max-samples` | Max samples per type | `--max-samples 100` |
| `-v` | Verbose output | `-v` |

### Results Processor Options  
| Option | Description | Example |
|--------|-------------|---------|
| `-i` | Input JSON file | `-i results.json` |
| `-o` | Output directory | `-o ./reports/` |
| `-t` | Data types to process | `-t answerable fake` |
| `-v` | Verbose output | `-v` |

## 📂 Data Types

- **answerable** - Non-conflict Querys
- **fake** - Factual Knowledge Conflict Querys
- **context_conflict** - Role Profile Conflict Querys
- **out_series** - Role Setting Conflict Querys
- **absent** - Absent Knowledge Conflict Querys

## 🔄 Complete Workflow

### Step 1: Build Evaluation Dataset
```bash
# From your experiment results
python3 construct_evaluation_data.py \
    -i "/path/to/experiment/results/*.json" \
    -o evaluation_prompts.jsonl \
    -m "your_model_name"
```

### Step 2: Get GPT Evaluations
```bash
# Send evaluation_prompts.jsonl to GPT and collect responses
# Save GPT responses as evaluation_results.json
```

### Step 3: Process Results  
```bash
# Analyze GPT evaluation responses
python3 get_evaluation_result.py \
    -i evaluation_results.json \
    -o ./analysis_reports/
```

## 🚀 Quick Examples

### Example 1: Full Pipeline with Default Paths
```bash
# Build datasets for all models
./build_dataset.sh --all

# Process results 
./run_evaluation.sh
```

### Example 2: Custom Processing
```bash
# Build specific model dataset
python3 construct_evaluation_data.py \
    -i "/data/llama_results/*.json" \
    -o llama_eval_dataset.jsonl \
    -m "llama-7b" \
    --max-samples 100

# Process with specific data types
python3 get_evaluation_result.py \
    -i gpt_evaluations.json \
    -o ./reports/ \
    -t answerable fake context_conflict \
    --verbose
```

## 🔧 File Structure
```
evaluation/
├── construct_evaluation_data.py         # Dataset construction tool
├── get_evaluation_result.py     # Results processing tool
├── bash/build_dataset.sh          # Dataset builder runner
├── bash/run_evaluation.sh         # Results processor runner  
├── prompt.py                 # Profile and template definitions
└── README.md                 # This documentation
```

## ⚠️ Requirements

- Python 3.6+
- Required packages: `pandas`, `openpyxl`, `tqdm`  
- `prompt.py` file with `PROFILE` and `EVALUATION_TEMPLATE` definitions

## 🐛 Troubleshooting

**Missing packages**: `pip install pandas openpyxl tqdm`

**Permission denied**: `chmod +x *.sh` 

**No files found**: Check file paths and use quotes for glob patterns

**Import errors**: Ensure `prompt.py` exists or tools will use fallback defaults
