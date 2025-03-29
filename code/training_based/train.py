
import fire
from datasets import Dataset
import torch
from transformers import  AutoTokenizer, TrainingArguments,AutoModelForCausalLM
from peft import PeftModel, LoraConfig,PrefixTuningConfig, get_peft_model
from trl import SFTTrainer
from transformers.trainer_callback import TrainerCallback
import os
import json
MAX_INPUT_LENGTH = 512
MAX_LENGTH = 2048
device_map = "auto"



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
        model.print_trainable_parameters()
        
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "right"


    def process_roleplay(example):
        all_text = [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        text = tokenizer.apply_chat_template(
           all_text, 
           tokenize=False, 
           add_generation_prompt=False
        )
        
        example["text"] = text
        example["text_length"] = len(tokenizer(example["text"]).input_ids)
        return example

    data = json.load(open(data_path,"r"))
    train_data = Dataset.from_list(data)
    train_data = train_data.map(process_roleplay,num_proc=8)
    print(train_data[0])
    train_data = train_data.filter(lambda x: x["text_length"] <= MAX_LENGTH)


    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir='logs',
        per_device_train_batch_size=1,
        gradient_accumulation_steps=128,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        # save_steps=1,
        num_train_epochs=num_train_epochs,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        warmup_ratio=0.1,
        report_to="none",  # 确保不使用外部工具，如 wandb
        logging_strategy="steps"  # 或者 "epoch"，根据您的需求
    )



    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        # peft_config=peft_config
        dataset_text_field="text",
        max_seq_length=MAX_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
        # callbacks=callbacks,
        callbacks= [],
    )

    trainer.train()
    model.save_pretrained(output_dir)
    output_dir = os.path.join(output_dir, "final_checkpoint")
    # if peft_type == "mulingpeft":
    #     model.save_model(os.path.join(output_dir, "delta_vector.pth"))
    # else:
    model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)