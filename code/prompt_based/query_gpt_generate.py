import sys
sys.path.append('/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/code')
# import torch
# import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
from prompt import QUERY_TEMPLATE,PROFILE,SIGNAL_QUERY_TEMPLATE
import os
import argparse
import logging

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_prompt_response(prompt, response):
    logger.info("Question:\n"+prompt)
    logger.info("Response:\n"+response)
    



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
    generated_ids = model.generate(encodes, max_new_tokens=1024)
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    if "[/INST]" in  decoded[0]:
        response = decoded[0].split("[/INST]")[-1]
    else:
        
        response = decoded[0].split("assistant\n")[-1]
    return response

def generate_response(
    model_path:str = "",
    data_path:str = "",
    save_path:str = "",
    query_type: str = "direct",
    ):
    dataset = load_dataset(data_path)
    query_dataset = []
    # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)   
    for role in tqdm(list(dataset.keys())):
        for key in tqdm(dataset[role]):
            # role_response[role][key] =[]
            for data in tqdm(dataset[role][key][:50]):
                question = data["question"]
                # if query_type == "direct":
                #     input_text = get_input_text(
                #         tokenizer=tokenizer,
                #         text=question,
                #         role=role,
                #         role_profile=PROFILE[role]
                #     )
                # elif query_type == "signal":
                #     input_text = get_input_text(
                #         tokenizer=tokenizer,
                #         text=question,
                #         role=role,
                #         role_profile=PROFILE[role],
                #         prompt_signal="True"
                #     )
                input_text = QUERY_TEMPLATE.format(question=question,role=role,role_profile=PROFILE[role])
                query_dataset.append({
                    "input_text":input_text,
                    "question":question,
                    "data_type":key,
                    "role":role,
                    "fake_method":data["fake_method"],
                })
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    save_path1 = save_path.replace(".jsonl","part1.jsonl")
    with open(save_path,"w",encoding="utf-8") as f:
        for data in query_dataset[:8000]:
            json.dump(data,f,ensure_ascii=False)
            f.write('\n')
    # save_path2 = save_path.replace(".jsonl","part2.jsonl")    
    # with open(save_path2,"w",encoding="utf-8") as f:
    #     for data in query_dataset[8000:]:
    #         json.dump(data,f,ensure_ascii=False)
    #         f.write('\n')
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate response using specified model and data.")
    parser.add_argument('--model_path', type=str, help='Path to the model')
    parser.add_argument('--data_path', type=str, default="/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/meta_data/test.json",help='Path to the data')
    parser.add_argument('--save_path', type=str, default="/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/evaluation/query/chatgpt_query.json",help='Path to save the generated response')
    parser.add_argument('--query_type', type=str, default="direct", help='Type of query')

    args = parser.parse_args()

    generate_response(
        model_path=args.model_path,
        data_path=args.data_path,
        save_path=args.save_path,
        query_type=args.query_type
    )