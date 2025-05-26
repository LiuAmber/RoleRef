# RoleRef: Enhancing Refusal Capabilities of Role-Playing Agents

## Overview

This repository contains the complete implementation for "Tell Me What You Don't Know: Enhancing Refusal Capabilities of Role-Playing Agents via Representation Space Analysis and Editing," which introduces a comprehensive framework for analyzing and improving how Role-Playing Agents (RPAs) handle conflicting or inappropriate queries.

Our research addresses a critical limitation in current RPAs: their tendency to provide direct responses to queries that conflict with their role knowledge rather than appropriately refusing to answer. Through systematic analysis of model representations and the development of novel intervention techniques, we demonstrate significant improvements in refusal capabilities while maintaining general role-playing performance.

## Key Contributions

The repository implements three primary methodological approaches for enhancing RPA refusal capabilities, along with comprehensive evaluation frameworks and datasets. Our representation editing approach achieves superior performance compared to traditional fine-tuning methods while requiring no additional model training.

## Repository Structure

### Core Implementation

**code/evaluation/** - Comprehensive evaluation framework and dataset construction tools
- `construct_evaluation_data.py` - Builds standardized evaluation datasets from roleplay experiments
- `get_evaluation_result.py` - Processes GPT evaluation responses and generates analysis reports  
- `bash/` - Automated scripts for dataset building and evaluation processing
- `README.md` - Detailed documentation for evaluation tools

**code/repe_based/** - Representation editing implementation and analysis tools
- `collect_activation.py` - Extracts model hidden state activations across different query types
- `classify.py` - Trains neural network classifiers for representation analysis
- `generate_with_representation_control.py` - Generates controlled responses through representation editing
- `rep_control_reading_vec.py` - Core framework for model wrapping and activation manipulation
- `bash/` - Execution scripts for different models and configurations
- `README.md` - Comprehensive documentation for representation analysis toolkit

**code/prompt_based/** - Prompt engineering approaches
- `generate.py` - Response generation using prompt-based refusal strategies
- `generate_question.py` - Systematic question generation for evaluation scenarios

**code/training_based/** - Supervised fine-tuning implementations
- `train.py` - Fine-tuning scripts for enhanced refusal capabilities
- `test.py` - Evaluation framework for trained models
- `generate.py` - Response generation from fine-tuned models

**code/job_code/** - Utility scripts for batch processing and job management

### Data and Results

**data/evaluation/** - Evaluation datasets and processed results
- `qwen_test_repe.json`, `llama_test_repe.json` - Model-specific evaluation datasets
- `processed/` - Comprehensive analysis reports in Excel format
- `result/` - Raw evaluation outputs from different experimental conditions

**data/generate/** - Generated responses and experimental outputs
- `prompt/qwen32/` - Extensive prompt-based generation results across 14 characters and 5 query types
- `representation/` - Controlled generation outputs using representation editing techniques

## Installation and Setup

### System Requirements

The framework requires Python 3.8 or higher with CUDA-compatible GPU support for optimal performance. Memory requirements vary by model size, with larger models requiring proportionally more GPU memory.

### Dependencies

```bash
pip install torch transformers pandas openpyxl tqdm numpy scikit-learn
pip install datasets accelerate peft
```

### Environment Configuration

Ensure CUDA environment variables are properly configured for multi-GPU setups. The representation editing framework automatically manages device placement for different model architectures.

## Usage Instructions

### Evaluation Pipeline

The evaluation framework provides end-to-end assessment of RPA refusal capabilities across multiple conflict scenarios.

**Dataset Construction:**
```bash
cd code/evaluation
python construct_evaluation_data.py -i "/path/to/experiment/results/*.json" -o evaluation_dataset.jsonl -m "model_name"
```

**Results Processing:**
```bash
python get_evaluation_result.py -i evaluation_results.json -o ./analysis_reports/ --verbose
```

**Automated Processing:**
```bash
bash bash/build_dataset.sh --all
bash bash/run_evaluation.sh
```

### Representation Analysis and Editing

The representation editing approach provides parameter-free enhancement of refusal capabilities through activation space manipulation.

**Activation Extraction:**
```bash
cd code/repe_based
python collect_activation.py -m /path/to/model -t ft -s ./activations -d questions.json
```

**Representation Classification:**
```bash
python classify.py --data_path ./activations --seed 42
```

**Controlled Generation:**
```bash
python generate_with_representation_control.py \
    --model_path /path/to/model \
    --data_path questions.json \
    --save_path ./controlled_responses \
    --ceoff -0.1
```

### Training-Based Approaches

For comparison with representation editing, the repository includes implementations of fine-tuning and LoRA-based enhancement methods.

**Fine-Tuning:**
```bash
cd code/training_based
python train.py --model_path /path/to/model --data_path /path/to/training/data
```

**Evaluation:**
```bash
python test.py --model_path /path/to/trained/model --evaluation_data /path/to/test/data
```

## Evaluation Framework

### RoleRef Benchmark

Our evaluation utilizes the RoleRef benchmark, which systematically assesses RPA capabilities across five distinct query categories. The benchmark encompasses 14 characters from popular novel series including Harry Potter, Lord of the Rings, Twilight, and The Hunger Games.

**Query Categories:**
- Non-conflict queries for baseline performance assessment
- Role setting conflicts involving cross-series contamination
- Role profile conflicts contradicting character descriptions  
- Factual knowledge conflicts containing false information
- Absent knowledge conflicts regarding events the character did not witness

### Evaluation Metrics

The framework employs nine comprehensive evaluation dimensions spanning general conversational ability, role-playing consistency, and refusal accuracy. Each dimension utilizes a three-point scoring system with detailed rubrics for consistent assessment.

**Assessment Dimensions:**
- Awareness of pitfalls and conflict recognition
- Refusal judgment accuracy and appropriateness
- Alignment with character background, style, abilities, and personality
- Response consistency, quality, and factual accuracy

## Research Findings

### Performance Analysis

Our comprehensive evaluation across state-of-the-art models reveals significant performance gaps between contextual knowledge conflicts and parametric knowledge conflicts. Models demonstrate strong capabilities in refusing contextual conflicts but struggle with parametric knowledge conflicts, achieving substantially lower scores in factual knowledge and absent knowledge scenarios.

### Representation Space Insights

Through linear probing and t-SNE visualization analysis, we identified distinct rejection regions and direct response regions within model representation spaces. This discovery explains the observed performance differences and provides the theoretical foundation for our representation editing approach.

### Method Comparison

Representation editing achieves superior balance between refusal capability enhancement and preservation of general role-playing abilities. While fine-tuning methods improve conflict query handling, they often degrade performance on non-conflict queries. Our approach maintains strong performance across all query types while requiring no additional model parameters.

## Model Compatibility

The framework supports major language model architectures including Llama series models, Qwen series models, Mistral variants, and GPT models. Model-specific configurations are automatically detected based on architecture specifications and hidden dimension requirements.

## Citation and Acknowledgments

If you utilize this work in your research, please cite our paper:

```bibtex
@article{liu2024roleref,
    title={Tell Me What You Don't Know: Enhancing Refusal Capabilities of Role-Playing Agents via Representation Space Analysis and Editing},
    author={Liu, Wenhao and An, Siyu and Lu, Junru and Wu, Muling and Li, Tianlong and Wang, Xiaohua and Lv, Changze and Zheng, Xiaoqing and Yin, Di and Sun, Xing and Huang, Xuanjing},
    journal={arXiv preprint},
    year={2024}
}
```

## Repository Access

Complete code and datasets are publicly available through our GitHub repository: https://github.com/LiuAmber/RoleRef

For questions regarding implementation details or research collaboration opportunities, please contact the corresponding authors through the provided institutional email addresses.

## License and Reproducibility

This research is provided under standard academic licensing terms. All experimental configurations and hyperparameters are documented to ensure reproducibility. We have conducted independent validation testing to verify that researchers can successfully replicate our findings using the provided codebase and documentation.