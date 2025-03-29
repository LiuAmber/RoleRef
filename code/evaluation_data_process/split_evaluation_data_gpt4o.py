import json

save_path = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/evaluation/cache/gpt4o.json"
data_paths = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/generate/query_gpt/gpt4o_result/roleplay_reject_answer_evaluation_gpt4o_*.txt"
import glob

data_paths = glob.glob(data_paths)

dataset = {}
for path in data_paths:
    with open(path,"r") as f:
        for line in f:
            # print(line)
            try:
                data = json.loads(line)
            except:
                print(line)
                continue
            role = data["role"]
            data_type = data["data_type"]
            if role not in dataset:
                dataset[role] = {}
            if data_type not in dataset[role]:
                dataset[role][data_type] = [data]
            else:
                dataset[role][data_type].append(data)
# print(dataset[role])
with open(save_path,"w") as f:
    for role in dataset:
        # print(role)
        for data_type in dataset[role]:
            print(role,data_type)
            for data in dataset[role][data_type][:50]:
                json.dump(data,f,ensure_ascii=False)
                f.write('\n')