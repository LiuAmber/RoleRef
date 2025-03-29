# import fire
# from datasets import Dataset
import torch
from transformers import  AutoTokenizer, TrainingArguments,AutoModelForCausalLM
from peft import PeftModel, LoraConfig,PrefixTuningConfig, get_peft_model
# from trl import SFTTrainer
from transformers.trainer_callback import TrainerCallback
import os
import json
MAX_INPUT_LENGTH = 512
MAX_LENGTH = 2048
device_map = "cpu"



def train(
        model_path: str = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/model/llama/3.1-8B-instruct",
        data_path: str = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/data/meta_data/train_dataset.json",
        output_dir: str = "/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/model",
        peft_type: str = "lora",
        learning_rate: float = 3e-4,
        num_train_epochs:  int = 2,
):
    print(f"model_path:{model_path}\n",
          f"data_path:{data_path}\n",
          f"output_dir:{output_dir}\n"
          )
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if peft_type == "ft":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            trust_remote_code=True,
            use_auth_token=True,
            torch_dtype=torch.bfloat16,
        )
    elif peft_type == "lora":
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        trust_remote_code=True,
        use_auth_token=True,
        torch_dtype=torch.bfloat16,
    )
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=16,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules = ["q_proj", "v_proj"]
        )
        model = get_peft_model(model, peft_config)
        print(model_path)
        model.print_trainable_parameters()

train(model_path="/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/model/llama/3-8B-instruct")
train(model_path="/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/model/llama/3.1-8B-instruct")
train(model_path="/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/model/mistral-7B")
train(model_path="/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/model/qwen2/7B-instruct")