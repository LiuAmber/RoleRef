import sys
sys.path.append('/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/code')
from transformers import AutoModelForCausalLM,LlamaForCausalLM,AutoTokenizer
from peft import AutoPeftModelForCausalLM
from generate_with_representation_control import wrap_model
# import datasets
import torch
import pickle
from tqdm import tqdm
import os
import json
import gc
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
# step1: load dataset

from prompt import QUERY_TEMPLATE,PROFILE,SIGNAL_QUERY_TEMPLATE

def get_prompt(tokenizer,text,role,role_profile,prompt_signal=None):
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

def get_rep(hs0):
    return [hs0[i][0, -1].detach().clone().cpu().numpy() for i in range(len(hs0))]

def get_response(model, tokenizer, prompt, max_new_tokens=1):
    try:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    except:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.model.device)
    # with torch.inference_mode():
    model.eval()    
    outputs = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, output_hidden_states=True, return_dict_in_generate=True)
    rep = get_rep(outputs.hidden_states[0])
    output_id = outputs.sequences[0][input_ids.shape[1]+1:]
    outs = tokenizer.decode(output_id, skip_special_tokens=True)
    torch.cuda.empty_cache()
    return (rep, outs)

def sort_list_and_indices(lst):
    x = sorted(range(len(lst)), key=lambda i: lst[i])
    return sorted(range(len(lst)), key=lambda i: lst[i]), sorted(lst)


def calculate_variance(numbers):
    mean = sum(numbers) / len(numbers)
    squared_diff_sum = sum((x - mean) ** 2 for x in numbers)
    variance = squared_diff_sum / len(numbers)
    return variance

def get_index_variance(diff_rep, layer_id):
    rep = [reps[layer_id] for reps in diff_rep]
    variance_list = []
    mean_list = []
    for i in range(len(rep[0])):
        selected = [rep[x][i] for x in range(len(rep))]
        mean = sum(selected) / len(selected) 
        mean_list.append(mean)
        variance = calculate_variance(selected)
        variance_list.append(variance)
    sorted_indices, sorted_list = sort_list_and_indices(variance_list)
    return sorted_indices, sorted_list, mean_list

def construct_common(sorted_indices, mean_list, num_selected=100,hidden_dim=4096):
    out = [0]*hidden_dim
    for i in range(num_selected):
        idx = sorted_indices[i]
        val = mean_list[idx]
        out[idx] = val
    return torch.tensor(out)

def construct_common_per_layer(diff_rep, layer_id, num_selected,hidden_dim=4096):
    sorted_indices, sorted_list, mean_list = get_index_variance(diff_rep, layer_id)
    common_rep_per_layer = construct_common(sorted_indices, mean_list, num_selected,hidden_dim)
    return common_rep_per_layer

def construct_all_common(diff_rep, num_selected,hidden_dim=4096):
    outs = []
    for i in tqdm(range(len(diff_rep[0]))):
        outs.append(construct_common_per_layer(diff_rep, i, num_selected,hidden_dim))
    return outs




name = {0:"answerable", 1:"out_series",2:"fake",3:"absent",4:"context_conflict"}


def get_activation_diff(model,tokenizer,i,save_path,role,role_profile,dataset):
    question_answer = dataset[role][name[0]][i]["question"]
    question_unanswer = dataset[role][name[1]][i]["question"]
    
    prompt_answer = get_prompt(tokenizer,question_answer,role,role_profile)
    prompt_unanswer = get_prompt(tokenizer,question_unanswer,role,role_profile)
    
    rep_answer, outs = get_response(model, tokenizer, prompt_answer)
    rep_unanswer, outs = get_response(model, tokenizer, prompt_unanswer)
    
    # save_path += f"/{i}.pt"
    diffs = []
    for j in range(len(rep_answer)):
        red = rep_answer[j]
        green = rep_unanswer[j]
        diff = red-green
        diffs.append(diff)
    # torch.save(diffs,save_path)
    try:
        question_fake = dataset[role][name[2]][i]["question"]
        prompt_fake = get_prompt(tokenizer,question_fake,role,role_profile)
        rep_fake, outs = get_response(model, tokenizer, prompt_fake)
    except:
        rep_fake=None
    try:
        question_absent = dataset[role][name[3]][i]["question"]
        prompt_absent = get_prompt(tokenizer,question_absent,role,role_profile)
        rep_absent, outs = get_response(model, tokenizer, prompt_absent)
    except:
        rep_absent=None
    try:
        question_context_conflict = dataset[role][name[4]][i]["question"]
        prompt_context_conflict = get_prompt(tokenizer,question_context_conflict,role,role_profile)
        rep_context_conflict, outs = get_response(model, tokenizer, prompt_context_conflict)
    except:
        rep_context_conflict=None
    return rep_answer,rep_unanswer,rep_fake,rep_absent,rep_context_conflict,diffs
    
def get_activations(
    model_name_or_path="/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/model/llama/3-8B-instruct",
    model_type = "ft",
    save_path = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/representation/llama3/",
    data_path = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/meta_data/train.json"
                    ):
    if "test" in data_path:
        end = 50
    else:
        end = -1
    print(model_name_or_path)
    print(data_path)
    print(save_path)
    if model_type == "ft":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="cuda")
    elif model_type == "lora":
        model = AutoPeftModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="cuda",
        trust_remote_code=True,
        use_auth_token=True,
    )
        
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", legacy=False)

    with open(data_path,"r") as f:
        dataset = json.load(f)
    
    for role in tqdm(dataset.keys()):
        if model_type == "repe":
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="cuda", trust_remote_code=True) 
            ceoff=-0.1
            layer_ids = [0,32]
            if "3.1-8B-instruct" in model_name_or_path:
                safe_pattern_path = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/representation/base/llama3.1/train/{role}/rep_diff_0.3.pkl"
            elif "3-8B-instruct" in model_name_or_path:
                safe_pattern_path = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/representation/base/llama3/train/{role}/rep_diff_0.3.pkl"
            elif "mistral-7B" in model_name_or_path:
                safe_pattern_path = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/representation/base/mistral/train/{role}/rep_diff_0.3.pkl"
            elif "qwen2" in model_name_or_path:
                safe_pattern_path = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/representation/base/qwen2/train/{role}/rep_diff_0.3.pkl"
                layer_ids = [0,28]
            
            safe_pattern_path = safe_pattern_path.format(role=role)
            model= wrap_model(
                        model,
                        tokenizer,
                    ceoff=ceoff,
                    layer_ids=list(range(layer_ids[0],layer_ids[1])),
                    safe_pattern_path=safe_pattern_path,
                    )
        role_save_path = f"{save_path}/{role}/"
        print(role)
        print(role_save_path)
        role_profile = PROFILE[role]
        
        if not os.path.exists(role_save_path):
            os.makedirs(role_save_path)
            
        rep_answer_list,rep_unanswer_list,diff_list = [],[],[]
        rep_fake_list,rep_absent_list,rep_context_conflict_list = [],[],[]
        for i in tqdm(range(len(dataset[role]["answerable"][:end]))):
            rep_answer,rep_unanswer,rep_fake,rep_absent,rep_context_conflict,diff = get_activation_diff(model,tokenizer,i,save_path,role,role_profile,dataset)
            rep_answer_list.append(rep_answer)
            rep_unanswer_list.append(rep_unanswer)
            diff_list.append(diff)
            if rep_fake != None:
                rep_fake_list.append(rep_fake) 
            if rep_absent != None:
                rep_absent_list.append(rep_absent) 
            if rep_context_conflict != None:
                rep_context_conflict_list.append(rep_context_conflict) 
        with open(f"{role_save_path}/rep_answer.pkl","wb") as f:
            pickle.dump(rep_answer_list,f)
        with open(f"{role_save_path}/rep_out_series.pkl","wb") as f:
            pickle.dump(rep_unanswer_list,f)
        with open(f"{role_save_path}/rep_fake.pkl","wb") as f:
            pickle.dump(rep_fake_list,f)
        with open(f"{role_save_path}/rep_absent.pkl","wb") as f:
            pickle.dump(rep_absent_list,f)
        with open(f"{role_save_path}/rep_context_conflict.pkl","wb") as f:
            pickle.dump(rep_context_conflict_list,f)
        with open(f"{role_save_path}/rep_diff.pkl","wb") as f:
            pickle.dump(diff_list,f)
        
        if model_type == "repe":
            hidden_size = model.model.config.hidden_size
        else:
            hidden_size = model.config.hidden_size
        num_selected = int(hidden_size * 0.5)
        out = construct_all_common(diff_list, num_selected=num_selected,hidden_dim=hidden_size)
        with open(f"{role_save_path}/rep_diff_half.pkl","wb") as f:
            pickle.dump(out,f)
        out = construct_all_common(diff_list, num_selected=hidden_size,hidden_dim=hidden_size)
        with open(f"{role_save_path}/rep_diff_all.pkl","wb") as f:
            pickle.dump(out,f)
        num_selected = int(hidden_size * 0.3)
        out = construct_all_common(diff_list, num_selected=num_selected,hidden_dim=hidden_size)
        with open(f"{role_save_path}/rep_diff_0.3.pkl","wb") as f:
            pickle.dump(out,f)
        if model_type == "repe":
            del model

import argparse



def main():
    parser = argparse.ArgumentParser(description="Get activations from a model.")
    parser.add_argument('--model_name_or_path', type=str, required=True,
                        help='Path to the model or model name.')
    parser.add_argument('--model_type', type=str, required=True,
                        help='Type of data (e.g., lora).')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save the activations.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the data file.')

    args = parser.parse_args()

    get_activations(
        model_name_or_path=args.model_name_or_path,
        model_type=args.model_type,
        save_path=args.save_path,
        data_path=args.data_path
    )

if __name__ == "__main__":
    main()




# def get_activations():
#     activations = {}
#     role = "Hermione Granger"
#     role_profile = PROFILE[role]
#     for id in tqdm(range(2)):
#         activation_dataset = []

#         questions = dataset[name[id]]
#         for question in tqdm(questions[:25]):
#             # print(question)
#             prompt = get_prompt(tokenizer,question,role,role_profile)
#             rep, outs = get_response(model, tokenizer, prompt)
#             print(f">>> Q-{question}\n>>> A:{outs}\n\n")
#             activation_dataset.append(rep)
        
#         print(f"\n\n\n\n")
#         if id == 0:
#             activations["answerable"] = activation_dataset
#         elif id == 1:
#             activations["unanswerable"] = activation_dataset
#     return activations

# activations = get_activations()
# diff_rep = []
# red_rep = activations["answerable"]
# green_rep = activations["unanswerable"]

# for i in tqdm(range(len(red_rep))):
#     diffs = []
#     reds = red_rep[i] 
#     greens = green_rep[i]
#     for j in range(len(reds)):
#         red = reds[j].to("cuda:0")
#         green = greens[j].to("cuda:0")
#         diff = red-green
#         diffs.append(diff)
#     diff_rep.append(diffs)
import glob
# model_name_or_path = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/model/llama3/8B-instruct"
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")

# use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
# tokenizer.pad_token_id = 0
diff_rep = []
# paths = glob.glob("/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/reject_answer/code/one_stage/representation/data/roleplay_test_activation/diff_single/*.pt")
# for path in paths :
#     diff_rep.append(torch.load(path, map_location=torch.device('cpu')))
# torch.save(diff_rep, f'/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/reject_answer/code/one_stage/representation/data/roleplay_test_activation/diff_rep_roleplay.pt')


# diff_rep += torch.load("/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/reject_answer/code/one_stage/representation/data/roleplay_test_activation/diff_rep.pt", map_location=torch.device('cpu'))

# ======================================================================= tianlong li method ======================================================================================================================================================

# data= torch.load("/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/reject_answer/code/one_stage/representation/data/roleplay_test_activation/diff_rep_roleplay.pt", map_location=torch.device('cpu'))
# def negate_nested_list(nested_list):
#     if isinstance(nested_list, list):
#         return [negate_nested_list(item) for item in nested_list]
#     elif isinstance(nested_list, torch.Tensor):
#         return -nested_list
#     else:
#         raise TypeError("Unsupported data type")
# negative_data = negate_nested_list(data)
# diff_rep += negative_data
# torch.save(diff_rep, f'/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/reject_answer/code/one_stage/representation/data/roleplay_test_activation/diff_rep_roleplay.pt')

# num_selected = model.config.hidden_size * 0.5
# out = construct_all_common(diff_rep, num_selected=model.config.hidden_size)
# torch.save(out, f'/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/reject_answer/code/one_stage/representation/data/roleplay_test_activation/common_diff_rep_half_concat.pt')


# num_selected = model.config.hidden_size
# out = construct_all_common(diff_rep, num_selected)
# torch.save(out, f'/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/reject_answer/code/one_stage/representation/data/roleplay_test_activation/common_diff_rep_all_concat.pt')


# num_selected = model.config.hidden_size * 0.3
# out = construct_all_common(diff_rep, num_selected=model.config.hidden_size)
# torch.save(out, f'/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/reject_answer/code/one_stage/representation/data/roleplay_test_activation/common_diff_rep_0.3_concat.pt')

# =================================================================== pca and kmeans =====================================================================================================================================================
# def pca(dataset):
#     out = []
#     for data in dataset:
#         data_matrix = np.array(data)
#         pca = PCA(n_components=1)
#         pca.fit(data_matrix)
#         # 提取第一个主成分向量
#         principal_component_vector = pca.components_[0]
#         # print(principal_component_vector.shape)
#         out.append(torch.Tensor(principal_component_vector))
#     return out
# transposed_data = list(map(list, zip(*diff_rep)))
# pca_out = pca(transposed_data)
# torch.save(pca_out,"/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/reject_answer/code/one_stage/representation/data/roleplay_test_activation/diff_rep_pca.pt")

# def kmean(dataset):
#     out = []
#     for data in dataset:
#         data_matrix = np.array(data)
#         kmeans = KMeans(n_clusters=1, random_state=0).fit(data_matrix)
#     # 获取聚类中心
#         cluster_center = kmeans.cluster_centers_[0]
#         # print(cluster_center.shape)
        
#         out.append(torch.Tensor(cluster_center))
#     return out
# transposed_data = list(map(list, zip(*diff_rep)))
# kmeans_out = kmean(transposed_data)
# torch.save(kmeans_out,"/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/reject_answer/code/one_stage/representation/data/roleplay_test_activation/diff_rep_kmeans.pt")
