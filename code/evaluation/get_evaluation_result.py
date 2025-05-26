#!/usr/bin/env python3
"""
Roleplay Evaluation Data Processing Script

This script processes evaluation results from roleplay experiments, extracting and analyzing
various metrics like awareness of pitfalls, refusal to answer judgment, role alignment, etc.
It can generate both detailed results (grouped by role) and aggregated results (across all roles).

Author: Wenhao Liu
Date: 2025
"""

import json
import glob
import argparse
import pandas as pd
import os
import sys
from typing import Dict, List, Any, Tuple

# Import prompt module (assumed to contain PROFILE dictionary)
try:
    from prompt import PROFILE
except ImportError:
    print("Warning: Could not import PROFILE from prompt module. Using empty dict.")
    PROFILE = {}


class RoleplayEvaluationProcessor:
    """
    Main class for processing roleplay evaluation data
    """
    
    def __init__(self, file_path: str, save_path: str, data_types: List[str] = None):
        """
        Initialize the processor with file paths and data types
        
        Args:
            file_path (str): Path to input JSON file(s) (supports glob patterns)
            save_path (str): Directory path to save output files
            data_types (list): List of data types to process
        """
        self.file_path = file_path
        self.save_path = save_path
        self.data_types = data_types or ["answerable", "out_series", "context_conflict", "fake", "absent"]
        
        # Define evaluation metrics columns
        self.columns = [
            'Role', 'question_type', 'Awareness_of_Pitfalls', 'Refusal_to_Answer_Judgment',
            'Alignment_with_Role_Background', 'Alignment_with_Role_Style', 
            'Alignment_with_Role_Abilities', 'Alignment_with_Role_Personality',
            'Consistency_of_Response', 'Quality_of_Response', 'Factuality_of_Response',
            'count', 'error_count'
        ]
        
        # Metrics to extract from GPT responses
        self.metrics = [
            'Awareness_of_Pitfalls', 'Refusal_to_Answer_Judgment', 'Alignment_with_Role_Background',
            'Alignment_with_Role_Style', 'Alignment_with_Role_Abilities', 'Alignment_with_Role_Personality',
            'Consistency_of_Response', 'Quality_of_Response', 'Factuality_of_Response'
        ]
        
        # Ensure save directory exists
        os.makedirs(self.save_path, exist_ok=True)
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        Load dataset from JSON file(s)
        
        Returns:
            List of dictionaries containing evaluation data
        """
        print(f"Loading dataset from: {self.file_path}")
        paths = glob.glob(self.file_path)
        
        if not paths:
            raise FileNotFoundError(f"No files found matching pattern: {self.file_path}")
        
        dataset = []
        for path in paths:
            print(f"Processing file: {path}")
            try:
                with open(path, "r", encoding='utf-8') as f:
                    data = json.load(f)
                    dataset.extend(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {path}: {e}")
                continue
            except Exception as e:
                print(f"Error reading file {path}: {e}")
                continue
        
        print(f"Loaded {len(dataset)} records from {len(paths)} file(s)")
        return dataset
    
    def extract_metrics_from_response(self, gpt_response: str) -> Tuple[Dict[str, float], bool]:
        """
        Extract evaluation metrics from GPT response text
        
        Args:
            gpt_response (str): The GPT evaluation response text
            
        Returns:
            Tuple of (metrics_dict, success_flag)
        """
        metrics = {}
        try:
            for metric in self.metrics:
                # Parse metric scores from GPT response
                metric_text = metric.replace('_', ' ')
                start_marker = f"{metric_text}:"
                end_marker = " point"
                
                if start_marker in gpt_response:
                    metric_part = gpt_response.split(start_marker)[1]
                    score_text = metric_part.split(end_marker)[0].strip()
                    metrics[metric] = float(score_text)
                else:
                    raise ValueError(f"Metric '{metric_text}' not found in response")
            
            return metrics, True
            
        except (ValueError, IndexError, AttributeError) as e:
            print(f"Error extracting metrics: {e}")
            return {}, False
    
    def process_data_for_role(self, dataset: List[Dict], role: str = None) -> Dict[str, Dict]:
        """
        Process evaluation data for a specific role or all roles
        
        Args:
            dataset (list): List of evaluation records
            role (str): Specific role to process, or None for all roles
            
        Returns:
            Dictionary with processed scores for each data type
        """
        final_scores = {}
        
        for data_type in self.data_types:
            print(f"Processing data type: {data_type}" + (f" for role: {role}" if role else " (all roles)"))
            
            # Initialize metric collections
            metric_collections = {metric: [] for metric in self.metrics}
            count = 0
            error_count = 0
            
            # Process each record in dataset
            for record in dataset:
                # Filter by role if specified
                if role and record.get("role") != role:
                    continue
                
                # Filter by data type
                if record.get("data_type") != data_type:
                    continue
                
                # Extract metrics from GPT response
                gpt_response = record.get("gpt_response", "")
                metrics, success = self.extract_metrics_from_response(gpt_response)
                
                if success:
                    # Add metrics to collections
                    for metric in self.metrics:
                        metric_collections[metric].append(metrics[metric])
                    count += 1
                else:
                    error_count += 1
            
            # Calculate average scores
            if count == 0:
                # No valid records found
                final_scores[data_type] = {metric: 0 for metric in self.metrics}
                final_scores[data_type].update({"count": count, "error_count": error_count})
            else:
                # Calculate averages and round to 4 decimal places
                final_scores[data_type] = {
                    metric: round(sum(metric_collections[metric]) / len(metric_collections[metric]), 4)
                    for metric in self.metrics
                }
                final_scores[data_type].update({"count": count, "error_count": error_count})
            
            print(f"  - Processed {count} records, {error_count} errors")
        
        return final_scores
    
    def create_dataframe_from_scores(self, all_scores: Dict[str, Dict[str, Dict]], role_name: str) -> pd.DataFrame:
        """
        Create pandas DataFrame from processed scores
        
        Args:
            all_scores (dict): Nested dictionary of scores
            role_name (str): Name of the role (or "all" for aggregated)
            
        Returns:
            pandas.DataFrame with formatted results
        """
        df = pd.DataFrame(columns=self.columns)
        
        for data_type, metrics in all_scores.items():
            # Add role and question type information
            metrics['Role'] = role_name
            metrics['question_type'] = data_type
            
            # Create temporary DataFrame and concatenate
            temp_df = pd.DataFrame([metrics])
            df = pd.concat([df, temp_df], ignore_index=True)
        
        return df
    
    def generate_detailed_results(self, dataset: List[Dict]) -> pd.DataFrame:
        """
        Generate detailed results grouped by role
        
        Args:
            dataset (list): List of evaluation records
            
        Returns:
            pandas.DataFrame with detailed results
        """
        print("\n=== Generating Detailed Results (by Role) ===")
        
        if not PROFILE:
            print("Warning: No PROFILE data available. Extracting roles from dataset...")
            roles = list(set(record.get("role", "") for record in dataset if record.get("role")))
        else:
            roles = list(PROFILE.keys())
        
        print(f"Processing {len(roles)} roles: {roles}")
        
        all_results = []
        
        for role in roles:
            print(f"\nProcessing role: {role}")
            role_scores = self.process_data_for_role(dataset, role)
            role_df = self.create_dataframe_from_scores(role_scores, role)
            all_results.append(role_df)
        
        # Combine all role results
        final_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame(columns=self.columns)
        
        return final_df
    
    def generate_aggregated_results(self, dataset: List[Dict]) -> pd.DataFrame:
        """
        Generate aggregated results across all roles
        
        Args:
            dataset (list): List of evaluation records
            
        Returns:
            pandas.DataFrame with aggregated results
        """
        print("\n=== Generating Aggregated Results (All Roles) ===")
        
        aggregated_scores = self.process_data_for_role(dataset, role=None)
        aggregated_df = self.create_dataframe_from_scores(aggregated_scores, "all")
        
        return aggregated_df
    
    def save_results(self, detailed_df: pd.DataFrame, aggregated_df: pd.DataFrame):
        """
        Save results to Excel files
        
        Args:
            detailed_df (pd.DataFrame): Detailed results by role
            aggregated_df (pd.DataFrame): Aggregated results
        """
        detailed_path = os.path.join(self.save_path, 'detail_result.xlsx')
        aggregated_path = os.path.join(self.save_path, 'all_result.xlsx')
        
        print(f"\nSaving detailed results to: {detailed_path}")
        detailed_df.to_excel(detailed_path, index=False)
        
        print(f"Saving aggregated results to: {aggregated_path}")
        aggregated_df.to_excel(aggregated_path, index=False)
        
        print("\nResults saved successfully!")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"Detailed results: {len(detailed_df)} rows")
        print(f"Aggregated results: {len(aggregated_df)} rows")
    
    def run(self):
        """
        Main execution method
        """
        try:
            # Load dataset
            dataset = self.load_dataset()
            
            # Generate detailed results
            detailed_df = self.generate_detailed_results(dataset)
            
            # Generate aggregated results
            aggregated_df = self.generate_aggregated_results(dataset)
            
            # Save results
            self.save_results(detailed_df, aggregated_df)
            
            return detailed_df, aggregated_df
            
        except Exception as e:
            print(f"Error during processing: {e}")
            sys.exit(1)


def main():
    """
    Main function to handle command line arguments and execute processing
    """
    parser = argparse.ArgumentParser(
        description="Process roleplay evaluation data and generate analysis reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python roleplay_evaluation.py -i /path/to/qwen_repe.json -o /path/to/output/

  # Specify custom data types
  python roleplay_evaluation.py -i /path/to/data.json -o /path/to/output/ -t answerable fake

  # Use glob pattern for multiple files
  python roleplay_evaluation.py -i "/path/to/data/*.json" -o /path/to/output/
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
        help='Output directory path to save Excel results'
    )
    
    parser.add_argument(
        '-t', '--types',
        nargs='+',
        default=["answerable", "out_series", "context_conflict", "fake", "absent"],
        help='Data types to process (default: answerable out_series context_conflict fake absent)'
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
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    if args.verbose:
        print(f"Input file pattern: {args.input}")
        print(f"Output directory: {args.output}")
        print(f"Data types to process: {args.types}")
    
    # Initialize and run processor
    processor = RoleplayEvaluationProcessor(
        file_path=args.input,
        save_path=args.output,
        data_types=args.types
    )
    
    detailed_df, aggregated_df = processor.run()
    
    if args.verbose:
        print("\n=== Detailed Results Preview ===")
        print(detailed_df.head())
        print("\n=== Aggregated Results Preview ===")
        print(aggregated_df.head())


if __name__ == "__main__":
    main()