# RoleRef: Enhancing Refusal Capabilities of Role-Playing Agents

## Overview

This repository contains the complete implementation for [Tell Me What You Don't Know: Enhancing Refusal Capabilities of Role-Playing Agents via Representation Space Analysis and Editing](https://arxiv.org/abs/2409.16913), which introduces a comprehensive framework for analyzing and improving how Role-Playing Agents (RPAs) handle conflicting or inappropriate queries.


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
- `generate.py` - Response generation from fine-tuned models

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

**Evaluation Dataset Construction:**
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

## Citation and Acknowledgments

If you utilize this work in your research, please cite our paper:

```bibtex
@misc{liu2024telldontknowenhancing,
      title={Tell Me What You Don't Know: Enhancing Refusal Capabilities of Role-Playing Agents via Representation Space Analysis and Editing}, 
      author={Wenhao Liu and Siyu An and Junru Lu and Muling Wu and Tianlong Li and Xiaohua Wang and Xiaoqing Zheng and Di Yin and Xing Sun and Xuanjing Huang},
      year={2024},
      eprint={2409.16913},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2409.16913}, 
}
```

## Repository Access

Complete code and datasets are publicly available through our GitHub repository: https://github.com/LiuAmber/RoleRef

For questions regarding implementation details or research collaboration opportunities, please contact the corresponding authors through the provided institutional email addresses.

## License and Reproducibility

This research is provided under standard academic licensing terms. All experimental configurations and hyperparameters are documented to ensure reproducibility. We have conducted independent validation testing to verify that researchers can successfully replicate our findings using the provided codebase and documentation.
