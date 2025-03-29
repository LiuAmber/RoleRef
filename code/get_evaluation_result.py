file_path = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/evaluation/cache/llama3_repe_3.1.json"
save_path = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/evaluation/result/llama3_repe_3.1_"
# part1 
import json
from prompt import *
import pandas as pd
columns = ['Role', 'question_type','Awareness_of_Pitfalls', 'Refusal_to_Answer_Judgment', 'Alignment_with_Role_Background',
           'Alignment_with_Role_Style', 'Alignment_with_Role_Abilities', 'Alignment_with_Role_Personality',
           'Consistency_of_Response', 'Quality_of_Response', 'Factuality_of_Response', 'count', 'error_count']
df = pd.DataFrame(columns=columns)


data_types = ["answerable","out_series","context_conflict","fake","absent"]
# data_types = ["answerable"]
import glob
paths = glob.glob(file_path)
dataset = []
for path in paths:
    with open(path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                dataset.append(obj)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

for role in PROFILE.keys():
    full_scores = {}
    final_scores = {}

    for key in data_types:
        cnt = 0
        error_cnt = 0
        Awareness_of_Pitfalls=[]
        Alignment_with_Role_Background=[]
        Alignment_with_Role_Style=[]
        Alignment_with_Role_Abilities=[]
        Alignment_with_Role_Personality=[]
        Consistency_of_Response=[]
        Quality_of_Response=[]
        Refusal_to_Answer_Judgment=[]
        Factuality_of_Response=[]
        for i in range(len(dataset)):
            if dataset[i]["role"] != role or dataset[i]["data_type"] != key:
                continue
            try:
                # if int(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"].split("Awareness of Pitfalls:")[1].split(" point")[0].strip()) > 1:
                # print(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"].split("Awareness of Pitfalls:")[1].split(" point"))
                Awareness_of_Pitfalls.append( float(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"].split("Awareness of Pitfalls:")[1].split(" point")[0].strip()))
                Refusal_to_Answer_Judgment.append( float(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"].split("Refusal to Answer Judgment:")[1].split(" point")[0].strip()))
                Alignment_with_Role_Background.append(float(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"].split("Alignment with Role Background:")[1].split(" point")[0].strip()))
                Alignment_with_Role_Style.append( float(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"].split("Alignment with Role Style:")[1].split(" point")[0].strip()))
                Alignment_with_Role_Abilities.append( float(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"].split("Alignment with Role Abilities:")[1].split(" point")[0].strip()))
                Alignment_with_Role_Personality.append(float(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"].split("Alignment with Role Personality:")[1].split(" point")[0].strip()))
                Consistency_of_Response.append(float(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"].split("Consistency of Response:")[1].split(" point")[0].strip()))
                Quality_of_Response.append( float(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"].split("Quality of Response:")[1].split(" point")[0].strip()))
                Factuality_of_Response.append( float(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"].split("Factuality of Response:")[1].split(" point")[0].strip()))

                cnt += 1
                # print(Awareness_of_Pitfalls,Awareness_of_Pitfalls/cnt)
            except:
                # print(i)
                # print(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"])
                error_cnt += 1
        if cnt == 0:
            final_scores[key] = {
            "Awareness_of_Pitfalls":0,
            "Refusal_to_Answer_Judgment":0,
            "Alignment_with_Role_Background":0,
            "Alignment_with_Role_Style":0,
            "Alignment_with_Role_Abilities":0,
            "Alignment_with_Role_Personality":0,
            "Consistency_of_Response":0,
            "Quality_of_Response":0,
            "Factuality_of_Response":0,
            "count":cnt,
        "error_count":error_cnt}
        else:
            final_scores[key] = {
            "Awareness_of_Pitfalls":round(sum(Awareness_of_Pitfalls)/len(Awareness_of_Pitfalls),4),
            "Refusal_to_Answer_Judgment":round(sum(Refusal_to_Answer_Judgment)/len(Refusal_to_Answer_Judgment),4),
            "Alignment_with_Role_Background":round(sum(Alignment_with_Role_Background)/len(Alignment_with_Role_Background),4),
            "Alignment_with_Role_Style":round(sum(Alignment_with_Role_Style)/len(Alignment_with_Role_Style),4),
            "Alignment_with_Role_Abilities":round(sum(Alignment_with_Role_Abilities)/len(Alignment_with_Role_Abilities),4),
            "Alignment_with_Role_Personality":round(sum(Alignment_with_Role_Personality)/len(Alignment_with_Role_Personality),4),
            "Consistency_of_Response":round(sum(Consistency_of_Response)/len(Consistency_of_Response),4),
            "Quality_of_Response":round(sum(Quality_of_Response)/len(Quality_of_Response),4),
            "Factuality_of_Response":round(sum(Factuality_of_Response)/len(Factuality_of_Response),4),
            "count":cnt,
            "error_count":error_cnt
        
    }
    for question_type, metrics in final_scores.items():
        metrics['Role'] = role
        metrics['question_type'] = question_type
        temp_df = pd.DataFrame([metrics])
        df = pd.concat([df, temp_df], ignore_index=True)

 
df.to_excel(save_path+'detail_result.xlsx', index=False)





# part2



from prompt import *
import pandas as pd
import json
columns = ['Role', 'question_type','Awareness_of_Pitfalls', 'Refusal_to_Answer_Judgment', 'Alignment_with_Role_Background',
           'Alignment_with_Role_Style', 'Alignment_with_Role_Abilities', 'Alignment_with_Role_Personality',
           'Consistency_of_Response', 'Quality_of_Response', 'Factuality_of_Response', 'count', 'error_count']
df = pd.DataFrame(columns=columns)


data_types = ["answerable","out_series","absent","fake","context_conflict"]
# data_types = ["answerable"]
import glob
paths = glob.glob(file_path)
dataset = []
for path in paths:
    with open(path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                dataset.append(obj)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")


    full_scores = {}
    final_scores = {}

for key in data_types:
    cnt = 0
    error_cnt = 0
    Awareness_of_Pitfalls=[]
    Alignment_with_Role_Background=[]
    Alignment_with_Role_Style=[]
    Alignment_with_Role_Abilities=[]
    Alignment_with_Role_Personality=[]
    Consistency_of_Response=[]
    Quality_of_Response=[]
    Refusal_to_Answer_Judgment=[]
    Factuality_of_Response=[]
    for i in range(len(dataset)):
        if dataset[i]["data_type"] != key:
            continue
        try:
            # if int(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"].split("Awareness of Pitfalls:")[1].split(" point")[0].strip()) > 1:
            # print(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"].split("Awareness of Pitfalls:")[1].split(" point"))
            Awareness_of_Pitfalls.append( float(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"].split("Awareness of Pitfalls:")[1].split(" point")[0].strip()))
            Refusal_to_Answer_Judgment.append( float(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"].split("Refusal to Answer Judgment:")[1].split(" point")[0].strip()))
            Alignment_with_Role_Background.append(float(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"].split("Alignment with Role Background:")[1].split(" point")[0].strip()))
            Alignment_with_Role_Style.append( float(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"].split("Alignment with Role Style:")[1].split(" point")[0].strip()))
            Alignment_with_Role_Abilities.append( float(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"].split("Alignment with Role Abilities:")[1].split(" point")[0].strip()))
            Alignment_with_Role_Personality.append(float(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"].split("Alignment with Role Personality:")[1].split(" point")[0].strip()))
            Consistency_of_Response.append(float(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"].split("Consistency of Response:")[1].split(" point")[0].strip()))
            Quality_of_Response.append( float(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"].split("Quality of Response:")[1].split(" point")[0].strip()))
            Factuality_of_Response.append( float(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"].split("Factuality of Response:")[1].split(" point")[0].strip()))

            cnt += 1
            # print(Awareness_of_Pitfalls,Awareness_of_Pitfalls/cnt)
        except:
            # print(i)
            # print(dataset[i]["answer_chatgpt4o"]["choices"][0]["message"]["content"])
            error_cnt += 1
    if cnt == 0:
        final_scores[key] = {
        "Awareness_of_Pitfalls":0,
        "Refusal_to_Answer_Judgment":0,
        "Alignment_with_Role_Background":0,
        "Alignment_with_Role_Style":0,
        "Alignment_with_Role_Abilities":0,
        "Alignment_with_Role_Personality":0,
        "Consistency_of_Response":0,
        "Quality_of_Response":0,
        "Factuality_of_Response":0,
        "count":cnt,
    "error_count":error_cnt}
    else:
        final_scores[key] = {
        "Awareness_of_Pitfalls":round(sum(Awareness_of_Pitfalls)/len(Awareness_of_Pitfalls),4),
        "Refusal_to_Answer_Judgment":round(sum(Refusal_to_Answer_Judgment)/len(Refusal_to_Answer_Judgment),4),
        "Alignment_with_Role_Background":round(sum(Alignment_with_Role_Background)/len(Alignment_with_Role_Background),4),
        "Alignment_with_Role_Style":round(sum(Alignment_with_Role_Style)/len(Alignment_with_Role_Style),4),
        "Alignment_with_Role_Abilities":round(sum(Alignment_with_Role_Abilities)/len(Alignment_with_Role_Abilities),4),
        "Alignment_with_Role_Personality":round(sum(Alignment_with_Role_Personality)/len(Alignment_with_Role_Personality),4),
        "Consistency_of_Response":round(sum(Consistency_of_Response)/len(Consistency_of_Response),4),
        "Quality_of_Response":round(sum(Quality_of_Response)/len(Quality_of_Response),4),
        "Factuality_of_Response":round(sum(Factuality_of_Response)/len(Factuality_of_Response),4),
        "count":cnt,
        "error_count":error_cnt
    
}
for question_type, metrics in final_scores.items():
    metrics['Role'] = "all"
    metrics['question_type'] = question_type
    temp_df = pd.DataFrame([metrics])
    df = pd.concat([df, temp_df], ignore_index=True)

df.to_excel(save_path+'all_result.xlsx', index=False)
print(df)