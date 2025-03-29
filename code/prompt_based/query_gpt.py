import openai
import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import fire
# from prompt import *
import logging

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
openai.api_key = "sk-nmfWP2siwRYTpHIA28893630C44f47Dd89Cb82F0B512E936"
openai.api_base = 'https://api.pumpkinaigc.online/v1'

def split_description():
    pass

def gpt_prompt(instruction):
    messages = [{"role": "system", "content": ""}]
    messages.append({"role": "user", "content": instruction})

    try:
        response = openai.ChatCompletion.create(
            # model="gpt-4o-mini",
            model = "gpt-3.5-turbo",
            # model="gpt-4",
            messages=messages,
            temperature=0,  # 值在[0,1]之间，越大表示回复越具有不确定性
            top_p=1,
            frequency_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            presence_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容,
            timeout=60)
    
        # 获取生成的文本
        generated_text = response['choices'][0]['message']['content']
        return generated_text

    except Exception as e:
        return str(e)

def process_question(question):
    query = question["input_text"]
    if "HTTP" in question["response"]:
        response = gpt_prompt(query)
        logger.info("question:\n"+question["question"])
        logger.info("response:\n"+response)
    # print(response)
        question["response"] = response
    return question

questions = json.load(open("/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/evaluation/cache/query/chatgpt/total.json", "r"))

res = []
temp_res = []
count = 0
batch_size = 1000

with ThreadPoolExecutor(max_workers=2) as executor:  # 你可以根据需要调整max_workers的数量
    futures = [executor.submit(process_question, question) for question in questions]
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        res.append(result)
        temp_res.append(result)
        count += 1


# Save all results to a final file
with open("/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/evaluation/cache/query/chatgpt_new.json", "w") as f:
    json.dump(res, f, indent=4,ensure_ascii=False)