import json

save_path = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/evaluation/cache/"
data_paths = [
    "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/generate/query_gpt/gpt4o_result/roleplay_reject_answer_evaluation_3_model_chatgpt_output_10140.txt"
]
dataset = {}
for path in data_paths:
    with open(path,"r") as f:
        for line in f:
            data = json.loads(line)
            model = data["model"]
            if model not in dataset:
                dataset[model] = [data]
            else:
                dataset[model].append(data)
for model in dataset:
    data_save_path = save_path + f"/{model}.json"
    with open(data_save_path,"w") as f:
        for data in dataset[model]:
            json.dump(data,f,ensure_ascii=False)
            f.write('\n')