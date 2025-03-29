import json

save_path = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/evaluation/cache/ft"
data_paths = [
    "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/evaluation/cache/temp_data/LLM_wjd_wlsjcj_ft_roleplay_reject_answer_evaluation_4_model_part1_20240902_6000/ft_roleplay_reject_answer_evaluation_4_model_part1_chatgpt_output_6000.txt",
    "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/evaluation/cache/temp_data/LLM_wjd_wlsjcj_ft_roleplay_reject_answer_evaluation_4_model_part2_20240902_7520/ft_roleplay_reject_answer_evaluation_4_model_part2_chatgpt_output_7520.txt"
]
count = 0
dataset = {}
models = ["llama3","llama3.1","mistral","qwen2"]
for path in data_paths:
    with open(path,"r") as f:
        for line in f:
            data = json.loads(line)
            # model = data["model"]
            index = count // 3380
            model = models[index]
            if model not in dataset:
                dataset[model] = [data]
            else:
                dataset[model].append(data)
            count += 1
for model in dataset:
    data_save_path = save_path + f"/{model}.json"
    with open(data_save_path,"w") as f:
        for data in dataset[model]:
            json.dump(data,f,ensure_ascii=False)
            f.write('\n')