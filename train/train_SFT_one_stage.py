from core.datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig, BitsAndBytesConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import argparse
import evaluate
import numpy as np
import os

attn_implementation = "flash_attention_2"
torch_dtype = torch.bfloat16

def process_func(example):
    MAX_LENGTH = 4096+2048
    input_ids, attention_mask, labels = [], [], []

    instruction = tokenizer(
        f"<|start_header_id|>user<|end_header_id|>\n\n{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
    response = tokenizer( f"{example['output']}<|eot_id|>", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    

    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]

    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def parse_args():
    parse = argparse.ArgumentParser(description='')
    parse.add_argument('--version', type=str, help='data version', default='c2_final_')
    parse.add_argument('--task', type=str, help='logiqa, math or ...', default='mbpp')
    parse.add_argument('--num_epochs', '-n', type=str, help='num_epochs', default='5')
    parse.add_argument('--epoch_delta', type=str, help='epoch delta', default=0)
    parse.add_argument('--resume', type=str, help='resume from checkpoint', default=False)
    parse.add_argument('--input_data', '-i', type=str, help='input data path',default="")
    parse.add_argument('--output', type=str, help='output path', default="")
    parse.add_argument('--model', '-m',  type=str, help='model full name', default="Meta-Llama-3-8B-Instruct")
    parse.add_argument('--model_path', '-mp', type=str, help='path to model', default='')
    args = parse.parse_args()
    return args

args = parse_args()
model = args.model
task = args.task
peft_model_id = "../model/" + args.version + model + '/' + task + args.num_epochs
save_path = "../model/" + args.version + model + '/' + task + str(int(args.num_epochs)+int(args.epoch_delta))
if args.model_path:
    model_path = args.model_path
else:
    model_path = f'/path/to/model/{model}'
if args.input_data:
    data_path = args.input_data
else:
    data_path = f'../data_train/{model}_{task}_train.jsonl'

if args.output: 
    path = args.output
else:
    path = f'../model/c2_{task}_{model}/'



df = pd.read_json(data_path, lines=True)

df['input'] = df['reflect_prompt'] + "\nUsing your reflection, generate a new answer to the question, your answer SHOULD not contain any reasoning. provide your answer in the format \"Answer: YOUR_ANSWER\""

df['output'] = df['reflections'] + " \n\n Answer: " + df['second_reason'].apply(lambda x: x.split("Observation:")[0]) 


ds = Dataset.from_pandas(df[['input', 'output']])

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

eval_df = pd.read_json(f'../data_train/{model}_{task}_eval.jsonl', lines=True)

eval_df['input'] = eval_df['reflect_prompt']

eval_df['output'] = eval_df['reflections'] + eval_df['second_reason'].apply(lambda x: x.split("Observation:")[0])
eval_ds=Dataset.from_pandas(eval_df[['input', 'output']])
eval_tokenized_id = eval_ds.map(process_func, remove_columns=eval_ds.column_names)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",  quantization_config=bnb_config, torch_dtype=torch.bfloat16)


config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)


if args.resume == 'True':
    model = PeftModel.from_pretrained(model, model_id = path +os.listdir(path)[0])
    model = model.merge_and_unload()
    num_train = int(args.num_epochs)
else:
    num_train = int(args.num_epochs)

model.enable_input_require_grads()
model = get_peft_model(model, config)

training_args = TrainingArguments(
    output_dir=path,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=num_train,
    save_strategy = "steps", 
    save_steps=50,
    learning_rate=1e-3,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",

    evaluation_strategy="steps",
    eval_steps=50,
    greater_is_better=False,
    load_best_model_at_end=True, 
    metric_for_best_model ="loss",
    per_device_eval_batch_size=8,
    save_total_limit = 2,
    )


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_id,
    eval_dataset=eval_tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),

)

if args.resume == 'True':
    trainer.train(resume_from_checkpoint = True)
else:
    trainer.train(resume_from_checkpoint = False)
trainer.save_model(save_path)
print(trainer.state.best_model_checkpoint)
