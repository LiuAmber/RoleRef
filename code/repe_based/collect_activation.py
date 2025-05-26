#!/usr/bin/env python3
"""
Model Activation Extractor for Roleplay Experiments

This script extracts model activations for different question types and computes
representation differences for analysis and potential representation editing.

Author: Wenhao Liu
Date: 2025
"""

import sys
import os
import json
import pickle
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM

import os
from pathlib import Path
current_dir = Path(__file__).parent
code_dir = current_dir / "code"  # Adjust relative path as needed
sys.path.append(str(code_dir))

try:
    from generate_with_representation_control import wrap_model
    from prompt import QUERY_TEMPLATE, PROFILE, SIGNAL_QUERY_TEMPLATE
except ImportError:
    print("Warning: Could not import representation control or prompt modules")
    QUERY_TEMPLATE = "Question: {question}\nRole: {role}\nProfile: {role_profile}\nAnswer:"
    PROFILE = {}
    SIGNAL_QUERY_TEMPLATE = QUERY_TEMPLATE


class ActivationExtractor:
    """
    Main class for extracting model activations across different question types
    """
    
    def __init__(self, model_name_or_path: str, model_type: str = "ft", 
                 save_path: str = "./activations", data_path: str = "./data.json", 
                 max_samples: int = 50):
        """
        Initialize the activation extractor
        
        Args:
            model_name_or_path (str): Path to model or model name
            model_type (str): Model type ('ft', 'lora', 'repe')
            save_path (str): Directory to save activations
            data_path (str): Path to input data JSON
            max_samples (int): Maximum samples to process per role
        """
        self.model_name_or_path = model_name_or_path
        self.model_type = model_type
        self.save_path = save_path
        self.data_path = data_path
        self.max_samples = max_samples
        
        # Question type mapping
        self.question_types = {
            0: "answerable", 
            1: "out_series", 
            2: "fake", 
            3: "absent", 
            4: "context_conflict"
        }
        
        self.model = None
        self.tokenizer = None
        
        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer based on model type"""
        print(f"Loading model: {self.model_name_or_path}")
        print(f"Model type: {self.model_type}")
        
        if self.model_type == "ft":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path, device_map="auto"
            )
        elif self.model_type == "lora":
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                device_map="cuda",
                trust_remote_code=True,
                use_auth_token=True,
            )
        elif self.model_type == "repe":
            # REPE model will be wrapped per role
            pass
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, padding_side="left", legacy=False
        )
    
    def get_prompt(self, text: str, role: str, role_profile: str, prompt_signal: str = None) -> str:
        """
        Create prompt for given text and role
        
        Args:
            text (str): Question text
            role (str): Character role
            role_profile (str): Role profile description
            prompt_signal (str): Optional prompt signal
            
        Returns:
            str: Formatted prompt
        """
        template = SIGNAL_QUERY_TEMPLATE if prompt_signal else QUERY_TEMPLATE
        
        messages = [{
            "role": "user", 
            "content": template.format(
                question=text, role=role, role_profile=role_profile,
                prompt_signal=prompt_signal or ""
            )
        }]
        
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    def get_response_and_activations(self, prompt: str, max_new_tokens: int = 1):
        """
        Get model response and hidden state activations
        
        Args:
            prompt (str): Input prompt
            max_new_tokens (int): Maximum tokens to generate
            
        Returns:
            tuple: (representations, output_text)
        """
        try:
            device = self.model.device
        except:
            device = self.model.model.device
        
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        self.model.eval()
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids, 
                max_new_tokens=max_new_tokens, 
                output_hidden_states=True, 
                return_dict_in_generate=True
            )
        
        # Extract representations from last token of each layer
        representations = [
            hidden_state[0, -1].detach().clone().cpu().numpy() 
            for hidden_state in outputs.hidden_states[0]
        ]
        
        # Extract generated text
        output_ids = outputs.sequences[0][input_ids.shape[1]+1:]
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        torch.cuda.empty_cache()
        return representations, output_text
    
    def get_activation_differences(self, role: str, role_profile: str, sample_data: dict) -> dict:
        """
        Get activations for all question types and compute differences
        
        Args:
            role (str): Character role
            role_profile (str): Role profile
            sample_data (dict): Sample data containing different question types
            
        Returns:
            dict: Dictionary containing all representations and differences
        """
        results = {}
        
        # Process base questions (answerable vs out_series)
        answerable_prompt = self.get_prompt(
            sample_data["answerable"]["question"], role, role_profile
        )
        out_series_prompt = self.get_prompt(
            sample_data["out_series"]["question"], role, role_profile
        )
        
        rep_answerable, _ = self.get_response_and_activations(answerable_prompt)
        rep_out_series, _ = self.get_response_and_activations(out_series_prompt)
        
        # Compute difference (answerable - out_series)
        diff = [
            rep_answerable[i] - rep_out_series[i] 
            for i in range(len(rep_answerable))
        ]
        
        results.update({
            "rep_answerable": rep_answerable,
            "rep_out_series": rep_out_series,
            "diff": diff
        })
        
        # Process other question types if available
        for qtype in ["fake", "absent", "context_conflict"]:
            if qtype in sample_data and sample_data[qtype]:
                prompt = self.get_prompt(
                    sample_data[qtype]["question"], role, role_profile
                )
                rep, _ = self.get_response_and_activations(prompt)
                results[f"rep_{qtype}"] = rep
        
        return results
    
    def calculate_variance_and_construct_common(self, diff_list: list, hidden_size: int) -> dict:
        """
        Calculate variance across differences and construct common representations
        
        Args:
            diff_list (list): List of difference vectors
            hidden_size (int): Hidden dimension size
            
        Returns:
            dict: Common representations at different selection ratios
        """
        common_reps = {}
        
        for ratio, name in [(0.3, "0.3"), (0.5, "half"), (1.0, "all")]:
            num_selected = int(hidden_size * ratio)
            layer_reps = []
            
            for layer_idx in range(len(diff_list[0])):
                # Get representations for this layer across all samples
                layer_diffs = [diff[layer_idx] for diff in diff_list]
                
                # Calculate variance for each dimension
                layer_array = np.array(layer_diffs)
                means = np.mean(layer_array, axis=0)
                variances = np.var(layer_array, axis=0)
                
                # Select dimensions with lowest variance
                selected_indices = np.argsort(variances)[:num_selected]
                
                # Construct common representation
                common_rep = np.zeros(hidden_size)
                common_rep[selected_indices] = means[selected_indices]
                layer_reps.append(torch.tensor(common_rep))
            
            common_reps[name] = layer_reps
        
        return common_reps
    
    def setup_repe_model(self, role: str):
        """
        Setup model with representation editing for specific role
        
        Args:
            role (str): Character role to setup REPE for
        """
        # Load base model for REPE
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, device_map="cuda", trust_remote_code=True
        )
        
        # Configure REPE parameters
        coeff = -0.1
        layer_ids = [0, self.model.config.num_hidden_layers]
        
        # Determine safe pattern path based on model type
        if "3.1-8B-instruct" in self.model_name_or_path:
            base_path = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/representation/base/llama3.1"
        elif "3-8B-instruct" in self.model_name_or_path:
            base_path = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/representation/base/llama3"
        elif "mistral-7B" in self.model_name_or_path:
            base_path = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/representation/base/mistral"
        elif "qwen2" in self.model_name_or_path:
            base_path = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/representation/base/qwen2"
        else:
            raise ValueError(f"Unsupported model for REPE: {self.model_name_or_path}")
        
        safe_pattern_path = f"{base_path}/train/{role}/rep_diff_0.3.pkl"
        
        # Wrap model with representation editing
        try:
            self.model = wrap_model(
                self.model,
                self.tokenizer,
                ceoff=coeff,
                layer_ids=list(range(layer_ids[0], layer_ids[1])),
                safe_pattern_path=safe_pattern_path,
            )
        except Exception as e:
            print(f"Warning: Could not setup REPE for {role}: {e}")
    
    def process_role(self, role: str, role_data: dict):
        """
        Process all samples for a specific role
        
        Args:
            role (str): Character role
            role_data (dict): Role data containing different question types
        """
        print(f"Processing role: {role}")
        
        # Setup REPE model if needed
        if self.model_type == "repe":
            self.setup_repe_model(role)
        
        role_profile = PROFILE.get(role, f"Character: {role}")
        role_save_path = os.path.join(self.save_path, role)
        os.makedirs(role_save_path, exist_ok=True)
        
        # Process samples
        all_results = {qtype: [] for qtype in ["answerable", "out_series", "fake", "absent", "context_conflict"]}
        diff_list = []
        
        max_samples = min(len(role_data["answerable"]), self.max_samples)
        
        for i in tqdm(range(max_samples), desc=f"Processing {role}"):
            try:
                # Prepare sample data
                sample_data = {}
                for qtype in self.question_types.values():
                    if qtype in role_data and i < len(role_data[qtype]):
                        sample_data[qtype] = role_data[qtype][i]
                
                # Get activations and differences
                results = self.get_activation_differences(role, role_profile, sample_data)
                
                # Store results
                for qtype in self.question_types.values():
                    if f"rep_{qtype}" in results:
                        all_results[qtype].append(results[f"rep_{qtype}"])
                
                if "diff" in results:
                    diff_list.append(results["diff"])
                    
            except Exception as e:
                print(f"Error processing sample {i} for role {role}: {e}")
                continue
        
        # Save all representations
        for qtype, reps in all_results.items():
            if reps:
                with open(os.path.join(role_save_path, f"rep_{qtype}.pkl"), "wb") as f:
                    pickle.dump(reps, f)
        
        # Save differences
        if diff_list:
            with open(os.path.join(role_save_path, "rep_diff.pkl"), "wb") as f:
                pickle.dump(diff_list, f)
            
            # Calculate and save common representations
            if self.model_type == "repe":
                hidden_size = self.model.model.config.hidden_size
            else:
                hidden_size = self.model.config.hidden_size
            
            common_reps = self.calculate_variance_and_construct_common(diff_list, hidden_size)
            
            for name, reps in common_reps.items():
                filename = f"rep_diff_{name}.pkl"
                with open(os.path.join(role_save_path, filename), "wb") as f:
                    pickle.dump(reps, f)
        
        # Clean up REPE model
        if self.model_type == "repe":
            del self.model
            torch.cuda.empty_cache()
        
        print(f"✓ Completed processing role: {role}")
    
    def extract_activations(self):
        """
        Main method to extract activations for all roles
        """
        print(f"Loading data from: {self.data_path}")
        
        # Load dataset
        with open(self.data_path, "r", encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Load model and tokenizer (except for REPE which loads per role)
        if self.model_type != "repe":
            self.load_model_and_tokenizer()
        else:
            # Only load tokenizer for REPE
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, padding_side="left", legacy=False
            )
        
        print(f"Processing {len(dataset)} roles")
        
        # Process each role
        for role in dataset.keys():
            try:
                self.process_role(role, dataset[role])
            except Exception as e:
                print(f"Error processing role {role}: {e}")
                continue
        
        print("✓ Activation extraction completed!")


def main():
    """
    Main function to handle command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Extract model activations for roleplay experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic fine-tuned model
  python3 activation_extractor.py -m /path/to/model -t ft -s ./activations -d data.json

  # LoRA model
  python3 activation_extractor.py -m /path/to/lora -t lora -s ./outputs

  # REPE model with custom samples
  python3 activation_extractor.py -m /path/to/model -t repe --max-samples 100
        """
    )
    
    parser.add_argument(
        '-m', '--model_name_or_path',
        required=True,
        help='Path to model or model name'
    )
    
    parser.add_argument(
        '-t', '--model_type',
        choices=['ft', 'lora', 'repe'],
        default='ft',
        help='Model type (default: ft)'
    )
    
    parser.add_argument(
        '-s', '--save_path',
        default='./activations',
        help='Directory to save activations (default: ./activations)'
    )
    
    parser.add_argument(
        '-d', '--data_path',
        default='./data.json',
        help='Path to input data JSON (default: ./data.json)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=50,
        help='Maximum samples to process per role (default: 50)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Model: {args.model_name_or_path}")
        print(f"Model type: {args.model_type}")
        print(f"Save path: {args.save_path}")
        print(f"Data path: {args.data_path}")
        print(f"Max samples: {args.max_samples}")
    
    # Initialize and run extractor
    extractor = ActivationExtractor(
        model_name_or_path=args.model_name_or_path,
        model_type=args.model_type,
        save_path=args.save_path,
        data_path=args.data_path,
        max_samples=args.max_samples
    )
    
    extractor.extract_activations()


if __name__ == "__main__":
    main()