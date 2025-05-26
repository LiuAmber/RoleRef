import sys
import os
from pathlib import Path

# Dynamically add the code directory to Python path
current_dir = Path(__file__).parent
code_dir = current_dir / "code"  # Adjust relative path as needed
sys.path.append(str(code_dir))

# Standard library imports
import json
import argparse
import logging

# Third-party imports
import torch
import transformers
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Local imports
from prompt import QUERY_TEMPLATE, PROFILE, SIGNAL_QUERY_TEMPLATE

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_prompt_response(prompt, response):
    logger.info("Question:\n"+prompt)
    logger.info("Response:\n"+response)
    
device_map="auto"


def load_dataset(path):
    with open(path,"r",encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset

def get_input_text(tokenizer,text,role,role_profile,prompt_signal=None):
    if prompt_signal == None:
        messages = [
            {"role": "user", "content": QUERY_TEMPLATE.format(question=text,role=role,role_profile=role_profile)}
        ]
    else:
        messages = [
            {"role": "user", "content": SIGNAL_QUERY_TEMPLATE.format(question=text,role=role,role_profile=role_profile,prompt_signal=prompt_signal)}
        ]
    input_text = tokenizer.apply_chat_template(
           messages, 
           tokenize=False, 
           add_generation_prompt=True
        )
    return input_text

def get_response(model, tokenizer, prompt):
    encodes = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    # print(encodes)
    generated_ids = model.generate(input_ids=encodes, max_new_tokens=1024)
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    if "[/INST]" in  decoded[0]:
        response = decoded[0].split("[/INST]")[-1]
    else:
        
        response = decoded[0].split("assistant\n")[-1]
    return response

def generate_response(
    model_path:str = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/model/checkpoint-18",
    peft_type:str = "lora",
    data_path:str = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/meta_data/test.json",
    save_path:str = "",
    query_type: str = "direct",
    ):
    dataset = load_dataset(data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    if peft_type == "lora":
        model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        trust_remote_code=True,
        use_auth_token=True,
    )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            trust_remote_code=True,
            use_auth_token=True,
        )
    if "llama3.1-instruct" in model_path:
        start = 10
    elif "llama3-instruct" in model_path:
        start = 13
    elif "mistral-instruct" in model_path:
        start = 8
    else:
        start = 0
    for role in tqdm(list(dataset.keys())[start:]):
        for key in tqdm(dataset[role]):
            role_response = {
                role:{}
            }
            role_response[role][key] =[]
            for data in tqdm(dataset[role][key][:50]):
                question = data["question"]
                if query_type == "direct":
                    input_text = get_input_text(
                        tokenizer=tokenizer,
                        text=question,
                        role=role,
                        role_profile=PROFILE[role]
                    )
                elif query_type == "signal":
                    input_text = get_input_text(
                        tokenizer=tokenizer,
                        text=question,
                        role=role,
                        role_profile=PROFILE[role],
                        prompt_signal="True"
                    )
                response = get_response(model, tokenizer, prompt=input_text)
                log_prompt_response(question, response)
                role_response[role][key].append({
                    "question":question,
                    "response":response,
                    "fake_method":data["fake_method"],
                })
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            role_save_path = save_path + f"/{role}_{key}.json"
            with open(role_save_path,"w",encoding="utf-8") as f:
                json.dump(role_response,f,indent=4,ensure_ascii=False)
                
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate response using specified model and data.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the generated response')
    parser.add_argument('--query_type', type=str, required=True, help='Type of query')
    parser.add_argument('--peft_type', type=str, required=True, help='Type of peft')

    args = parser.parse_args()

    generate_response(
        model_path=args.model_path,
        data_path=args.data_path,
        save_path=args.save_path,
        query_type=args.query_type,
        peft_type=args.peft_type
    )