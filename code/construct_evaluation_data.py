import json
import glob
from prompt import *
from tqdm import tqdm
def construct_evaluation_dataset(data_path,save_path,model_name):
    paths = glob.glob(data_path)
    # model_name = data_path.split("/")[-3]
    dataset = {}
    query_dataset = []
    for path in paths:
        with open(path,"r") as f:
            data = json.load(f)
            role = list(data.keys())[0]
            if role not in dataset:
                dataset.update(data)
            else:
                dataset[role].update(data[role])
    for role in dataset:
        # print(dataset)
        for key in dataset[role]:
            for i in tqdm(range(len(dataset[role][key][:50]))):
                question = dataset[role][key][i]["question"]
                answer = dataset[role][key][i]["response"]
                role_profile = PROFILE[role]
                if key == "answerable":
                    question_pitfalls="There are no pitfalls in the question"
                elif key == "out_series":
                    question_pitfalls="This is a question from another series of novels."
                elif key == "absent":
                    question_pitfalls="The character was absent at the time in question"
                elif key == "fake":
                    fake_method = dataset[role][key][i]["fake_method"]
                    question_pitfalls=f"This is a fake question and it conflicts with character knowledge or abilities, it replaces the concept:{fake_method}"
                elif key == "context_conflict":
                    fake_method = dataset[role][key][i]["fake_method"]
                    question_pitfalls=f"This is a fake question and it conflicts with Character Description, it replaces the concept:{fake_method}"
                dimention = "full_dimentions"
                prompt=EVALUATION_TEMPLATE[dimention].format(
                    question=question,
                    role=role,
                    role_profile=role_profile,
                    question_pitfalls=question_pitfalls,
                    answer = answer,
                )
                query_dataset.append({
                    "question":prompt,
                    "role":role,
                    "data_type":key,
                    "fake_method":dataset[role][key][i]["fake_method"],
                    "model":model_name,
                })

    print(len(query_dataset))
    with open(save_path, 'w') as file:
        # 将字典转换为JSON字符串并写入文件，每个字典占一行
        for data in tqdm(query_dataset):
            json.dump(data, file,ensure_ascii=False)
            file.write('\n')

def construct_evaluation_dataset_gpt(data_path,save_path,model_name):
    paths = glob.glob(data_path)
    # model_name = data_path.split("/")[-3]
    dataset = []
    query_dataset = []
    for path in paths:
        with open(path,"r") as f:
            data = json.load(f)
            dataset.extend(data)

    for data in dataset:

        question = data["question"]
        answer = data["response"]
        role = data["role"]
        key = data["data_type"]
        role_profile = PROFILE[role]
        if key == "answerable":
            question_pitfalls="There are no pitfalls in the question"
        elif key == "out_series":
            question_pitfalls="This is a question from another series of novels."
        elif key == "absent":
            question_pitfalls="The character was absent at the time in question"
        elif key == "fake":
            fake_method = data["fake_method"]
            question_pitfalls=f"This is a fake question and it conflicts with character knowledge or abilities, it replaces the concept:{fake_method}"
        elif key == "context_conflict":
            fake_method = data["fake_method"]
            question_pitfalls=f"This is a fake question and it conflicts with Character Description, it replaces the concept:{fake_method}"
        dimention = "full_dimentions"
        prompt=EVALUATION_TEMPLATE[dimention].format(
            question=question,
            role=data["role"],
            role_profile=role_profile,
            question_pitfalls=question_pitfalls,
            answer = answer,
        )
        query_dataset.append({
            "question":prompt,
            "role":role,
            "data_type":key,
            "fake_method":data["fake_method"],
            "model":model_name,
        })

    print(len(query_dataset))
    with open(save_path, 'w') as file:
        # 将字典转换为JSON字符串并写入文件，每个字典占一行
        for data in tqdm(query_dataset):
            json.dump(data, file,ensure_ascii=False)
            file.write('\n')

# construct_evaluation_dataset(
#     data_path="/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/generate/test/llm/llama_3-72B-instruction/*.json",
#     save_path = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/evaluation/query/roleplay_reject_answer_evaluation_prompt_llama3-72B-instruct.json",
#     model_name = "llama-3-72B-instruct"
# )
# construct_evaluation_dataset(
#     data_path="/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/generate/test/llm/llama_3.1-72B-instruction/*.json",
#     save_path = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/evaluation/query/roleplay_reject_answer_evaluation_prompt_llama3.1-72B-instruct.json",
#     model_name = "llama-3.1-72B-instruct"
# )
# construct_evaluation_dataset(
#     data_path="/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/generate/test/llm/mistxal-8x7B-instruct/*.json",
#     save_path = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/evaluation/query/roleplay_reject_answer_evaluation_prompt_mistral-8x7B-instruct.json",
#     model_name = "mistral-8x7B"
# )
# construct_evaluation_dataset(
#     data_path="/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/generate/test/llm/qwen2-72B-instruct/*.json",
#     save_path = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/evaluation/query/roleplay_reject_answer_evaluation_prompt_qwen2-72B-instruct.json",
#     model_name = "qwen2-72B-instruct"
# )



construct_evaluation_dataset(
    data_path="/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/generate/test/llama/3-8B-instruct/reprentation_merge3.1/*.json",
    save_path = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/evaluation/query/roleplay_reject_answer_evaluation_representation_llama3_repe_3.1.json",
    model_name = "llama3_repe_3.1"
)
# construct_evaluation_dataset_gpt(
#     data_path="/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/evaluation/cache/query/mini_new.json",
#     save_path = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/evaluation/query/roleplay_reject_answer_evaluation_prompt_gpt4o-mini.json",
#     model_name = "gpt4o-mini"
# )