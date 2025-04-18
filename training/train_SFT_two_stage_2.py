from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig, BitsAndBytesConfig, EarlyStoppingCallback, IntervalStrategy
import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import argparse
import evaluate
import numpy as np
import os
import json
import deepspeed
from accelerate import infer_auto_device_map
import torch.nn as nn

project_path = # TODO: Replace here with your project absolute path
os.environ['WANDB_PROJECT'] = 'reflection-training'
os.environ['WANDB_LOG_MODEL'] = 'checkpoint'
os.environ['WANDB_WATCH'] = 'all'
os.environ['WANDB_SILENT'] = 'False'
os.environ['WANDB_CACHE_DIR'] = # TODO: Replace here with your cache folder, if you prefer to use the default dir, delete this line
os.environ['TOKENIZERS_PARALLELISM'] = "true"
attn_implementation = "flash_attention_2"
torch_dtype = torch.bfloat16

def process_func(example):
    MAX_LENGTH = 4096+2048    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    # add_special_tokens 不在开头加 special_tokens
    instruction = tokenizer(
        f"<|start_header_id|>user<|end_header_id|>\n\n{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
    response = tokenizer( f"{example['output']}<|eot_id|>", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    
    # 因为eos token咱们也是要关注的所以 补充为1
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]

    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:  # 做一个截断
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
    parse.add_argument('--version', type=str, help='data version')
    parse.add_argument('--task', type=str, help='gsm or hotpotqa')
    parse.add_argument('--num_epochs', type=str, help='num_epochs')
    parse.add_argument('--epoch_delta', type=str, help='epoch delta', default="5")
    parse.add_argument('--resume', type=str, help='resume from checkpoint')
    parse.add_argument('--output', type=str, help='output file path')
    parse.add_argument('--model_path', type=str, help='base model path', default="Meta-Llama-3-8B-Instructs")
    parse.add_argument('--template', type=str, help='template', default="1")
    parse.add_argument('--checkpoint_num', type=str, help='checkpoint number', default="100")
    parse.add_argument('--bs', type=int, help='batch size', default=4)
    parse.add_argument('--ebs', type=int, help='eval and save steps', default=30)
    parse.add_argument('--ss', type=str, help='save strategy', default="steps")
    parse.add_argument('--lr', type=float, help='learning rate', default=0.00001)
    parse.add_argument('--folder', type=str, help='data folder location', default="data_train")
    parse.add_argument('--wd', type=float, help='weight decay', default=0.0)
    parse.add_argument('--wr', type=float, help='warmup ratio', default=0.0)
    parse.add_argument('--gas', type=int, help='gradient accumulation steps', default=4)
    parse.add_argument('--checkpoint_path', type=str, help='checkpoint path', default="")
    args = parse.parse_args()
    return args


args = parse_args()
attn_implementation = "eager" if "gemma" in args.model_path else "flash_attention_2"

print("==========Loading Tokenizer and Models==========")
model_path = # TODO: Replace your model path here, 
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
path = # TODO: Replace here with your path to checkpoints if args.checkpoint_path == "" else args.checkpoint_path
if args.resume == 'True':
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation=attn_implementation)
    num_train = int(args.epoch_delta)
    print("loading from a checkpoint")
    print(path)
else:
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16, attn_implementation=attn_implementation)
    num_train = int(args.num_epochs)

model.enable_input_require_grads()
print(f"Training with model {args.model_path} and resume from checkpoint {args.resume}")

print("==========Loading Dataset==========")

print(f"Training with task {args.task} at output {args.folder}")
print("Training with first stage of Reflection")
# data_path = f'{project_path}/data/{args.folder}/{args.model_path}_{args.task}_train.jsonl'
data_path = f'{project_path}/data/{args.folder}/{args.model_path}_{args.task}_train.jsonl'
# 逐行读取 JSONL 文件并构建 DataFrame
df = pd.read_json(data_path, lines=True)  # 这里要改成其他路径
# 提取需要的字段并重命名为 input 和 output
df['input'] = df['reflect_prompt']
# 对每一行的 'scratchpad' 进行分割并获取需要的部分
df['output'] = df['reflections']

# 创建Hugging Face的数据集
ds = Dataset.from_pandas(df[['input', 'output']])
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

# breakpoint()
eval_path = f'{project_path}/data/{args.folder}/{args.model_path}_{args.task}_eval.jsonl'
eval_df = pd.read_json(eval_path, lines=True)
# 提取需要的字段并重命名为 input 和 output
eval_df['input'] = eval_df['reflect_prompt']
# 对每一行的 'scratchpad' 进行分割并获取需要的部分
eval_df['output'] = eval_df['reflections']
eval_ds=Dataset.from_pandas(eval_df[['input', 'output']])
# eval_ds = Dataset.from_pandas(pd.read_json('./data/' + args.version + args.task.replace("train","eval") + '.jsonl'))
eval_tokenized_id = eval_ds.map(process_func, remove_columns=eval_ds.column_names)

# import pdb; pdb.set_trace()
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch_dtype,
#     bnb_4bit_use_double_quant=True,
# )
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",  quantization_config=bnb_config, torch_dtype=torch.bfloat16)

'''
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj"],  #"o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1  # Dropout 比例
)
'''


# model = get_peft_model(model, config)


def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    # logits, labels = eval_preds
    # predictions = np.argmax(logits, axis=-1)
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions[0]
    # import pdb; pdb.set_trace()
    return metric.compute(predictions=pred_ids.flatten(), references=labels_ids.flatten())
'''
    logits = torch.tensor(eval_preds.predictions[0], dtype=torch.bfloat16)
    labels = torch.tensor(eval_preds.label_ids,  dtype=torch.long)

    # Typical shape for seq2seq or token-level tasks: (batch_size, seq_len, vocab_size)
    # Flatten so that CrossEntropyLoss sees (N, vocab_size) vs (N) for labels.
    logits = logits.view(-1, logits.size(-1))  # shape: (batch_size * seq_len, vocab_size)
    labels = labels.view(-1)                   # shape: (batch_size * seq_len)

    # Define the cross-entropy loss function
    loss_fct = nn.CrossEntropyLoss()

    # Compute the cross entropy
    loss = loss_fct(logits, labels)

    # Return as a dictionary for the Trainer
    return {"cross_entropy": loss.item()}
'''
def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels


print("==========Loading Training Args==========")
training_args = TrainingArguments(
    output_dir="/mnt/buffer/wangquansen/"+args.output,
    per_device_train_batch_size=args.bs,
    gradient_accumulation_steps=args.gas,
    logging_steps=1,
    num_train_epochs=num_train,
    save_strategy = args.ss, 
    save_steps=args.ebs,
    learning_rate=args.lr,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
    eval_strategy=args.ss,
    eval_steps=args.ebs,
    greater_is_better=False, # load model with highest F1 score
    load_best_model_at_end=True, 
    metric_for_best_model ="loss",
    per_device_eval_batch_size=args.bs,
    save_total_limit = 3,
    run_name = "reflection-training-"+args.output+"-"+args.num_epochs,
    weight_decay=args.wd,
    deepspeed="./ds_z3_offload_config.json",
    logging_dir="./logs",
    bf16=True,
    # resume_from_checkpoint=True,
    warmup_ratio=args.wr,
)
# weight_decay = 0.1
print("==========Loading Trainer==========")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_id,
    eval_dataset=eval_tokenized_id,
    # compute_metrics=compute_metrics,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=1e-4)]
)

# print("==========Training begin==========")
print(f"==========Trainer is in model parallel: {trainer.is_model_parallel}==========")
print(f"Training with model {args.model_path} and resume from checkpoint {args.resume}")
print(f"Training with task {args.task} at output {args.folder}")
print(f"attention implementation {attn_implementation}")
print(f"batch size {args.bs}")
print(f"saving strategy {args.ss}")
print(f"learning rate {args.lr}")
print(f"data folder {args.folder}")
if args.resume: print(f"checkpoint {args.checkpoint_num}")
'''
if args.resume == 'True':
    trainer.train(resume_from_checkpoint = True)
else:
    trainer.train(resume_from_checkpoint = False)
'''
print("==========Training begin==========")
trainer.train(resume_from_checkpoint = False)
#trainer.train()
print("==========Training complete, saving best model==========")
trainer.save_model(os.path.join(args.output, "best_model"))
print(trainer.state.best_model_checkpoint)

print("==========Training Ends==========")
