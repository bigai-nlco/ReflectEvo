from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig, BitsAndBytesConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import argparse
import evaluate
import numpy as np
import os
from dataHelper import get_dataset
from ..prompts.prompts_SFT import (            # 这里有修改！！！！！！改成有ans的refl了
    REASON_PROMPT_SFT,
    LOGIQA_FORMAT,
    MATH_FORMAT,
    MBPP_FORMAT,
    BIGBENCH_FORMAT,
    BIGBENCH_FREE_FORMAT,
)
from ..prompts.fewshots_SFT import (
    LOGIQA_FEWSHOTS_SFT,
    MATH_FEWSHOTS_SFT,
    MBPP_FEWSHOTS_SFT,
    BIGBENCH_FEWSHOTS_SFT,
    BIGBENCH_FREE_FEWSHOTS_SFT, #bigbench freetext
)
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
    parse.add_argument('--version', type=str, help='data version', default='new0207_c2_final_')   # 和下一项拼接，最终存model的地方
    parse.add_argument('--task', '-t', type=str, help='logiqa, math or ...', default='mbpp')
    parse.add_argument('--num_epochs', '-n', type=str, help='num_epochs', default='5')
    parse.add_argument('--epoch_delta', type=str, help='epoch delta', default=0)        # 如果是继续训练（resume=True），还要训练几轮
    parse.add_argument('--resume', type=str, help='resume from checkpoint', default=False)
    parse.add_argument('--input_data', '-i', type=str, help='input data path',default="")
    parse.add_argument('--output', type=str, help='resume from checkpoint', default="")   # 这里的model就不放在git目录下了。这里是训练的checkpoint存放的地方
    parse.add_argument('--model', '-m', type=str, help='model full name', default="Meta-Llama-3-8B-Instruct")
    parse.add_argument('--model_path', '-mp', type=str, help='path to model', default='')
    args = parse.parse_args()
    return args

args = parse_args()
model = args.model
task = args.task
peft_model_id = "../model/new0207_" + args.version + model + '_SFT/' + task + args.num_epochs
save_path = "../model/new0207_" + args.version + model + '/' + task + str(int(args.num_epochs)+int(args.epoch_delta))
if args.model_path:
    model_path = args.model_path
else:
    model_path = f'/path/to/model/{model}'
if args.input_data:
    data_path = args.input_data
else:
    data_path = f'../data_train/{model}_{task}_train.jsonl'
# 这里的path是ckpt path
if args.output: 
    path = args.output
else:
    path = f'../../model/new0207_c2_{task}_{model}_SFT/'
# 将JSON文件转换为CSV文件
# 根据任务类型设置 input_format 和 input_fewshots
LOGIQA_OUTPUT_FORMAT="1 word"
MATH_OUTPUT_FORMAT="20 words"
MBPP_OUTPUT_FORMAT="150 words"
BIGBENCH_OUTPUT_FORMAT="1 word"
BIGBENCH_FREE_OUTPUT_FORMAT="30 words"

def get_task_config(task):
    if task == "logiqa":
        return LOGIQA_FORMAT, LOGIQA_FEWSHOTS_SFT, LOGIQA_OUTPUT_FORMAT
    elif task == "math":
        return MATH_FORMAT, MATH_FEWSHOTS_SFT, MATH_OUTPUT_FORMAT
    elif task == "mbpp":
        return MBPP_FORMAT, MBPP_FEWSHOTS_SFT, MBPP_OUTPUT_FORMAT
    elif task == "bigbench":
        return BIGBENCH_FORMAT, BIGBENCH_FEWSHOTS_SFT, BIGBENCH_OUTPUT_FORMAT
    elif task == "bigbenchfree":
        return BIGBENCH_FREE_FORMAT, BIGBENCH_FREE_FEWSHOTS_SFT, BIGBENCH_FREE_OUTPUT_FORMAT
    else:
        raise ValueError(f"Unknown task: {task}")
input_format, input_fewshots, output_format = get_task_config(task)
def cuda_memory():
    if torch.cuda.is_available():
        # 获取当前 GPU 的显存使用情况
        allocated = torch.cuda.memory_allocated()  # 当前分配的显存（字节）
        reserved = torch.cuda.memory_reserved()    # 当前保留的显存（字节）
        total = torch.cuda.get_device_properties(0).total_memory  # GPU 总显存（字节）

        # 转换为 GiB
        allocated_gib = allocated / 1024**3
        reserved_gib = reserved / 1024**3
        total_gib = total / 1024**3
        free_gib = total_gib - (allocated_gib + reserved_gib)

        # 输出显存使用情况
        print(f"GPU 0 总显存: {total_gib:.2f} GiB")
        print(f"当前分配的显存: {allocated_gib:.2f} GiB")
        print(f"当前保留的显存: {reserved_gib:.2f} GiB")
        print(f"可用显存: {free_gib:.2f} GiB")
    else:
        print("没有可用的 GPU")

cuda_memory()

df = pd.DataFrame(get_dataset(task, train_eval = 'train'))
eval_df = pd.DataFrame(get_dataset(task, train_eval = 'eval'))

print("数据集输入示例: ", REASON_PROMPT_SFT.format(question='问题', format=input_format, examples=input_fewshots))
df['input'] = df['question'].apply(lambda x: REASON_PROMPT_SFT.format(question=x, format=output_format, examples=input_fewshots))
df['output'] = df['answer']

# 创建Hugging Face的数据集
ds = Dataset.from_pandas(df[['input', 'output']])
print(ds[0])
cuda_memory()

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

eval_df['input'] = eval_df['question'].apply(lambda x: REASON_PROMPT_SFT.format(question=x, format=output_format, examples=input_fewshots))
eval_df['output'] = eval_df['answer']
eval_ds=Dataset.from_pandas(eval_df[['input', 'output']])
print('------------------------------------------------')
print(eval_ds[0])
eval_tokenized_id = eval_ds.map(process_func, remove_columns=eval_ds.column_names)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",  quantization_config=bnb_config, torch_dtype=torch.bfloat16)

cuda_memory()

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj"],  #"o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1  # Dropout 比例
)


if args.resume == 'True':
    model = PeftModel.from_pretrained(model, model_id = path +os.listdir(path)[0]) #path +os.listdir(path)[0],  +args.version + args.task + args.num_epochs
    model = model.merge_and_unload()
    num_train = int(args.num_epochs)
else:
    num_train = int(args.num_epochs)

model.enable_input_require_grads()
model = get_peft_model(model, config)

training_args = TrainingArguments(
    output_dir=path,    # 这里以及把保存的目录改为RFL_new下
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    logging_steps=10,   # 这里和下面的save_step分别调大
    num_train_epochs=num_train,
    save_strategy = "steps",
    save_steps=10,
    learning_rate=5e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",

    evaluation_strategy="steps",
    eval_steps=10,
    greater_is_better=False, # load model with highest F1 score
    load_best_model_at_end=True, 
    metric_for_best_model ="loss",
    per_device_eval_batch_size=8,
    save_total_limit = 2,
    )


cuda_memory()
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_id,
    eval_dataset=eval_tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, max_length=200),     # 这里要改！！！！！！
)
cuda_memory()

if args.resume == 'True':
    trainer.train(resume_from_checkpoint = True)
else:
    trainer.train(resume_from_checkpoint = False)
trainer.save_model(save_path)
