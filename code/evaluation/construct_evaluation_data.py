#!/usr/bin/env python3
"""
Evaluation Dataset Builder for Roleplay Experiments

This script constructs evaluation datasets from roleplay experiment results.
It processes JSON files containing character responses and creates standardized
evaluation prompts for GPT assessment.

Author: Wenhao Liu
Date: 2025
"""

import json
import glob
import argparse
import os
import sys
from typing import Dict, List, Any
from tqdm import tqdm

# Import prompt module (assumed to contain PROFILE and EVALUATION_TEMPLATE)
try:
    from prompt import PROFILE, EVALUATION_TEMPLATE
except ImportError:
    print("Warning: Could not import PROFILE and EVALUATION_TEMPLATE from prompt module.")
    PROFILE = {}
    EVALUATION_TEMPLATE = {}


class EvaluationDatasetBuilder:
    """
    Main class for building evaluation datasets from roleplay experiment results
    """
    
    def __init__(self, data_path: str, save_path: str, model_name: str, max_samples: int = 50):
        """
        Initialize the dataset builder
        
        Args:
            data_path (str): Path to input JSON files (supports glob patterns)
            save_path (str): Output file path for JSONL dataset
            model_name (str): Name of the model being evaluated
            max_samples (int): Maximum samples per role-datatype combination
        """
        self.data_path = data_path
        self.save_path = save_path
        self.model_name = model_name
        self.max_samples = max_samples
        
        # Define question pitfall descriptions for different data types
        self.pitfall_descriptions = {
            "answerable": "There are no pitfalls in the question",
            "out_series": "This is a question from another series of novels.",
            "absent": "The character was absent at the time in question",
            "fake": "This is a fake question and it conflicts with character knowledge or abilities, it replaces the concept:{fake_method}",
            "context_conflict": "This is a fake question and it conflicts with Character Description, it replaces the concept:{fake_method}"
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
    
    def detect_data_format(self, data: Any) -> str:
        """
        Detect the format of input data (nested or flat)
        
        Args:
            data: JSON data to analyze
            
        Returns:
            str: 'nested' for role->datatype->list structure, 'flat' for list structure
        """
        if isinstance(data, list):
            return 'flat'
        elif isinstance(data, dict):
            # Check if it's a nested structure (role -> datatype -> list)
            for role_key, role_data in data.items():
                if isinstance(role_data, dict):
                    return 'nested'
            return 'flat'
        else:
            raise ValueError("Unknown data format")
    
    def get_pitfall_description(self, data_type: str, fake_method: str = "") -> str:
        """
        Get pitfall description for a given data type
        
        Args:
            data_type (str): Type of data (answerable, fake, etc.)
            fake_method (str): Fake method description for fake/context_conflict types
            
        Returns:
            str: Formatted pitfall description
        """
        template = self.pitfall_descriptions.get(data_type, "Unknown data type")
        
        if "{fake_method}" in template and fake_method:
            return template.format(fake_method=fake_method)
        return template
    
    def create_evaluation_prompt(self, question: str, role: str, answer: str, 
                               data_type: str, fake_method: str = "") -> str:
        """
        Create evaluation prompt using the template
        
        Args:
            question (str): Original question
            role (str): Character role name
            answer (str): Model's response
            data_type (str): Type of question
            fake_method (str): Fake method if applicable
            
        Returns:
            str: Formatted evaluation prompt
        """
        role_profile = PROFILE.get(role, f"Character: {role}")
        question_pitfalls = self.get_pitfall_description(data_type, fake_method)
        
        # Use full_dimensions template or create a simple one if not available
        template_key = "full_dimentions"  # Note: keeping original typo for compatibility
        if template_key not in EVALUATION_TEMPLATE:
            template_key = "full_dimensions"  # Try corrected spelling
        
        if template_key in EVALUATION_TEMPLATE:
            prompt = EVALUATION_TEMPLATE[template_key].format(
                question=question,
                role=role,
                role_profile=role_profile,
                question_pitfalls=question_pitfalls,
                answer=answer,
            )
        else:
            # Fallback template if EVALUATION_TEMPLATE is not available
            prompt = f"""
Evaluate the following roleplay response:

Question: {question}
Role: {role}
Role Profile: {role_profile}
Question Pitfalls: {question_pitfalls}
Answer: {answer}

Please evaluate this response across multiple dimensions.
"""
        
        return prompt
    
    def process_nested_format(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process nested data format (role -> datatype -> list of samples)
        
        Args:
            data (dict): Nested data structure
            
        Returns:
            List of evaluation dataset entries
        """
        query_dataset = []
        
        for role in data:
            print(f"Processing role: {role}")
            role_data = data[role]
            
            for data_type in role_data:
                print(f"  Processing data type: {data_type}")
                samples = role_data[data_type][:self.max_samples]
                
                for i, sample in enumerate(tqdm(samples, desc=f"{role}-{data_type}")):
                    try:
                        question = sample["question"]
                        answer = sample["response"]
                        fake_method = sample.get("fake_method", "")
                        
                        prompt = self.create_evaluation_prompt(
                            question, role, answer, data_type, fake_method
                        )
                        
                        query_dataset.append({
                            "question": prompt,
                            "role": role,
                            "data_type": data_type,
                            "fake_method": fake_method,
                            "model": self.model_name,
                        })
                        
                    except KeyError as e:
                        print(f"Warning: Missing key {e} in sample {i} for {role}-{data_type}")
                        continue
        
        return query_dataset
    
    def process_flat_format(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process flat data format (list of samples with role/data_type fields)
        
        Args:
            data (list): List of data samples
            
        Returns:
            List of evaluation dataset entries
        """
        query_dataset = []
        
        print(f"Processing {len(data)} samples")
        
        for sample in tqdm(data, desc="Processing samples"):
            try:
                question = sample["question"]
                answer = sample["response"]
                role = sample["role"]
                data_type = sample["data_type"]
                fake_method = sample.get("fake_method", "")
                
                prompt = self.create_evaluation_prompt(
                    question, role, answer, data_type, fake_method
                )
                
                query_dataset.append({
                    "question": prompt,
                    "role": role,
                    "data_type": data_type,
                    "fake_method": fake_method,
                    "model": self.model_name,
                })
                
            except KeyError as e:
                print(f"Warning: Missing key {e} in sample")
                continue
        
        return query_dataset
    
    def load_and_merge_files(self) -> Dict[str, Any]:
        """
        Load and merge multiple JSON files
        
        Returns:
            Merged data structure
        """
        paths = glob.glob(self.data_path)
        
        if not paths:
            raise FileNotFoundError(f"No files found matching pattern: {self.data_path}")
        
        print(f"Found {len(paths)} files to process")
        
        # Try to determine format from first file
        with open(paths[0], "r", encoding='utf-8') as f:
            first_data = json.load(f)
        
        data_format = self.detect_data_format(first_data)
        print(f"Detected data format: {data_format}")
        
        if data_format == 'nested':
            # Merge nested structures
            merged_data = {}
            for path in paths:
                print(f"Loading file: {path}")
                with open(path, "r", encoding='utf-8') as f:
                    file_data = json.load(f)
                    
                    # Merge role data
                    for role in file_data:
                        if role not in merged_data:
                            merged_data[role] = file_data[role]
                        else:
                            # Merge data types within role
                            for data_type in file_data[role]:
                                if data_type not in merged_data[role]:
                                    merged_data[role][data_type] = file_data[role][data_type]
                                else:
                                    merged_data[role][data_type].extend(file_data[role][data_type])
            
            return merged_data
            
        else:  # flat format
            # Concatenate all lists
            merged_data = []
            for path in paths:
                print(f"Loading file: {path}")
                with open(path, "r", encoding='utf-8') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        merged_data.extend(file_data)
                    else:
                        print(f"Warning: Expected list in {path}, got {type(file_data)}")
            
            return merged_data
    
    def save_dataset(self, query_dataset: List[Dict[str, Any]]):
        """
        Save dataset in JSONL format
        
        Args:
            query_dataset (list): List of evaluation entries to save
        """
        print(f"Saving {len(query_dataset)} entries to: {self.save_path}")
        
        with open(self.save_path, 'w', encoding='utf-8') as file:
            for data in tqdm(query_dataset, desc="Saving"):
                json.dump(data, file, ensure_ascii=False)
                file.write('\n')
        
        print(f"✓ Dataset saved successfully!")
        print(f"Total entries: {len(query_dataset)}")
    
    def build_dataset(self):
        """
        Main method to build the evaluation dataset
        """
        try:
            # Load and merge input files
            merged_data = self.load_and_merge_files()
            
            # Detect format and process accordingly
            data_format = self.detect_data_format(merged_data)
            
            if data_format == 'nested':
                query_dataset = self.process_nested_format(merged_data)
            else:
                query_dataset = self.process_flat_format(merged_data)
            
            # Save the dataset
            self.save_dataset(query_dataset)
            
            return query_dataset
            
        except Exception as e:
            print(f"Error during dataset building: {e}")
            sys.exit(1)


def main():
    """
    Main function to handle command line arguments and execute building
    """
    parser = argparse.ArgumentParser(
        description="Build evaluation datasets from roleplay experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python3 construct_evaluation_data.py -i "/path/to/data/*.json" -o output.jsonl -m "llama-7b"

  # With custom sample limit
  python3 construct_evaluation_data.py -i input.json -o output.jsonl -m "qwen-7b" --max-samples 100

  # Process single file
  python3 construct_evaluation_data.py -i single_file.json -o dataset.jsonl -m "model_name"
        """
    )
    
    parser.add_argument(
        '-i', '--input', 
        required=True,
        help='Path to input JSON file(s). Supports glob patterns (e.g., "/path/to/*.json")'
    )
    
    parser.add_argument(
        '-o', '--output', 
        required=True,
        help='Output file path for JSONL dataset'
    )
    
    parser.add_argument(
        '-m', '--model', 
        required=True,
        help='Model name identifier'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=50,
        help='Maximum samples per role-datatype combination (default: 50)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate input file path
    if not glob.glob(args.input):
        print(f"Error: No files found matching pattern: {args.input}")
        sys.exit(1)
    
    if args.verbose:
        print(f"Input file pattern: {args.input}")
        print(f"Output file: {args.output}")
        print(f"Model name: {args.model}")
        print(f"Max samples per combination: {args.max_samples}")
    
    # Initialize and run builder
    builder = EvaluationDatasetBuilder(
        data_path=args.input,
        save_path=args.output,
        model_name=args.model,
        max_samples=args.max_samples
    )
    
    dataset = builder.build_dataset()
    
    if args.verbose:
        print(f"\n=== Build Summary ===")
        print(f"Total entries created: {len(dataset)}")
        
        # Show distribution by role and data type
        role_counts = {}
        type_counts = {}
        
        for entry in dataset:
            role = entry['role']
            data_type = entry['data_type']
            
            role_counts[role] = role_counts.get(role, 0) + 1
            type_counts[data_type] = type_counts.get(data_type, 0) + 1
        
        print(f"\nEntries by role:")
        for role, count in sorted(role_counts.items()):
            print(f"  {role}: {count}")
        
        print(f"\nEntries by data type:")
        for data_type, count in sorted(type_counts.items()):
            print(f"  {data_type}: {count}")


if __name__ == "__main__":
    main()