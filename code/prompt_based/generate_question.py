import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = """
### Task Description
Generate 10 questions unrelated to Gandalf's world. These questions should be impossible for Gandalf to answer, prompting him to reply with \"I don't know\".

### Requirements
1. Ensure the questions are diverse, covering different domains and topics.
2. Avoid generating duplicate or similar questions.
3. Questions should be beyond Gandalf's knowledge or unrelated to Gandalf's world.
4. Each question should be concise and clear.

### Output Format
Please return the questions you generated in the form of JSON.
Format Example:
```
["How...","What...",...]
```
"""

NON_CONFLICT_PROMPT = """### Task Description
Please act as a user who is having a conversation with Gandalf. You need to generate 10 questions related to Gandalf that will be used to ask him.

### Requirements
1. Ensure the authenticity of the questions, meaning they should be consistent with Gandalf's background, history, abilities, or experiences in the story.
2. Ensure diversity among the questions, covering different themes and aspects.
3. Avoid generating repetitive or overly similar questions.
4. The questions should be deep and capable of sparking meaningful answers.

### Return Format
Please return the questions you generated in the form of JSON.
Format Example:
```
["How...","What...",...]
```
"""

import re

def extract_code(text):
    pattern = r'```(?:json)?(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

def generate_questions(model_path, save_path, temperature=0.7, num_samples=5, max_retries=5):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

    messages = [{"role": "system", "content": NON_CONFLICT_PROMPT}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    generated_questions = []

    for i in range(num_samples):
        retries = 0
        while retries < max_retries:
            inputs = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
            
            generated_ids = model.generate(
                inputs, 
                max_new_tokens=1024,
                do_sample=True,
                temperature=temperature
            )
            decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)[len(inputs[0]):]

            try:
                questions = decoded_output
                generated_questions.append(questions)
                break
            except Exception as e:
                retries += 1
                print(f"Retry {retries}/{max_retries} due to parsing error: {e}")

        if retries == max_retries:
            print(f"Failed to generate questions after {max_retries} retries.")

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(generated_questions, f, indent=4, ensure_ascii=False)

    print(f"Questions saved to {save_path}")


if __name__ == "__main__":
    model_path = "/mnt/nj-1/dataset/liuwenhao/model/Qwen/Qwen2.5-7B-Instruct"
    save_path = "./generated_questions_qwen1.json"

    generate_questions(model_path, save_path)
