import sys
sys.path.append('/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/code')
import torch
import rep_control_reading_vec
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
from prompt import QUERY_TEMPLATE,PROFILE,SIGNAL_QUERY_TEMPLATE
import os
import argparse
import logging
import pickle
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
    encodes = tokenizer.encode(prompt, return_tensors="pt").to(model.model.device)
    generated_ids = model.generate(input_ids=encodes, max_new_tokens=1024)
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    if "[/INST]" in  decoded[0]:
        response = decoded[0].split("[/INST]")[-1]
    else:
        
        response = decoded[0].split("assistant\n")[-1]
    return response

def wrap_model(model,tokenizer,
               ceoff=0.1,
               layer_ids=list(range(0,32)),
               safe_pattern_path="/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/reject_answer/code/one_stage/representation/data/roleplay_test_activation/common_diff_rep_half_roleplay.pt"
               ):
    # layer_ids = 
    warpped_model = rep_control_reading_vec.WrappedReadingVecModel(model,tokenizer)
    warpped_model.reset()
    warpped_model.wrap_block(
        layer_ids=layer_ids,
        block_name="decoder_block"
    )
    with open(safe_pattern_path,"rb") as f:
        safe_pattern = pickle.load(f)
    safe_pattern = torch.stack(safe_pattern).to(dtype=torch.float32)
    # safe_pattern = torch.load(safe_pattern_path)
    activations={}

    for layer in layer_ids:
        activations[layer] = ceoff*safe_pattern[layer].to(warpped_model.model.device)
    warpped_model.set_controller(
        layer_ids=layer_ids, 
        activations=activations,
        block_name='decoder_block',
        # token_pos="end", 
        masks=None,
        normalize=False, 
        operator='linear_comb'
    )
    return warpped_model

def generate_response(
    model_path:str = "",
    data_path:str = "",
    save_path:str = "",
    query_type: str = "direct",
    layer_ids:list = [0,32],
    ceoff:float = -0.1,
    safe_pattern_path:str = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/reject_answer/code/one_stage/representation/data/roleplay/{role}/rep_diff_half.pkl",
    ):
    if "qwen" in model_path:
        layer_ids[1] = 28
    dataset = load_dataset(data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    start = 0
    for role in tqdm(list(dataset.keys())[start:]):
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)   
        safe_pattern_path = safe_pattern_path.format(role=role)
        model= wrap_model(
                    model,
                    tokenizer,
                ceoff=ceoff,
                layer_ids=list(range(layer_ids[0],layer_ids[1])),
                safe_pattern_path=safe_pattern_path,
                )
        for key in tqdm(dataset[role]):
            role_response = {
                role:{}
            }
            role_response[role][key] =[]
            for data in tqdm(dataset[role][key][:50]):
            # for data in tqdm(dataset[role][key][:1]):
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
        del model
        # break
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate response using specified model and data.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the generated response')
    parser.add_argument('--query_type', type=str, required=True, help='Type of query')
    parser.add_argument('--layer_ids', type=list,default=[0,32], help='Type of query')
    parser.add_argument('--ceoff', type=float,default=-0.1, help='Type of query')
    parser.add_argument('--safe_pattern_path', type=str, required=True, help='Type of query')
    
    args = parser.parse_args()

    generate_response(
        model_path=args.model_path,
        data_path=args.data_path,
        save_path=args.save_path,
        query_type=args.query_type,
        layer_ids=args.layer_ids,
        ceoff=args.ceoff,
        safe_pattern_path=args.safe_pattern_path
    )