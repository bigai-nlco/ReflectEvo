from os import environ
import pynvml
import json
import yaml
import sys
from core.dataHelper import get_dataset
from prompts.prompts import (
    REASON_PROMPT,
    LOGIQA_FORMAT,
    MATH_FORMAT,
    MBPP_FORMAT,
    BIGBENCH_FORMAT,
    BIGBENCH_FREE_FORMAT,
    REFLECTION_PROMPT,
    REFLECTION_PROMPT_TEST,
    DEMAND_TYPES,
    INT_TO_DEMAND_TYPES,
    DEMAND_TYPES_TO_INT
)
from prompts.fewshots import (
    HOTPOTQA_FEWSHOTS,
    LOGIQA_FEWSHOTS,
    MATH_FEWSHOTS,
    MBPP_FEWSHOTS,
    BIGBENCH_FEWSHOTS,
    BIGBENCH_FREE_FEWSHOTS
)
from envs.env_hotpotqa import HotPotQAEnv
from envs.env_logiqa import LogiQAEnv
from envs.env_mbpp import MBPPEnv
from envs.env_math import MATHEnv
from envs.env_bigbench import BigbenchEnv
from envs.env_bigbench_free import BigbenchfreeEnv
from llms import VLLMGenerator, make_generator
from core.new_agents import BatchCOTReflectAgent, BatchReactReflectAgent
import time


try:
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        print(f"GPU {i}: {name}")
except pynvml.NVMLError as err:
    print(f"Failed to initialize NVML: {err}")
    exit(1)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--method", "-m", type=str, help="COT or ReAct", default="COT")
parser.add_argument("--dataset", "-d", type=str, help="LogiQA, MATH, MBPP, Bigbench or Bigbenchfree")
parser.add_argument("--existing_dataset", "-ed", type=str, help="Regenerate it with the data generated earlier. Fill in the file path here")
parser.add_argument("--use_first_answer", "-u", type=str, help="Whether to keep the first answer in existing dataset", default="true")
parser.add_argument("--use_second_answer", "-us", type=str, help="Whether to keep the second answer in existing dataset", default="false")
parser.add_argument("--num_of_data", "-n", type=int, help="number of data, 0 for all data", default=0)
parser.add_argument("--demand_type", "-dt", type=int, help="The type of demand for the reflection task. Choose from 1 to 32.", default=1)
parser.add_argument("--output_file", "-o", type=str, help="output directory", required=False)
parser.add_argument("--model_name", "-mn", type=str, help="model name", default='Meta-Llama-3-8B-Instruct')
parser.add_argument("--reflection", "-r", type=str, help="HOW to use refl,1:regenerate reflection，2:use pre-stored reflection from existing dataset，3: Skip reflection generation", default="1")
parser.add_argument("--model_config", "-mc", type=str, help="YAML file address for model configuration.", default="")
parser.add_argument("--is_test", "-t", type=str, default="False")
parser.add_argument("--use_scratchpad", "-usc", type=str, help="Whether to use scrathpad in prompt",default="True")
parser.add_argument("--setting", "-s", type=str, default="")
args = parser.parse_args()
args.use_first_answer = args.use_first_answer.lower() == "true"
use_second_answer = args.use_second_answer.lower() == "true"
METHOD = args.method
if args.dataset:
    DATASET = args.dataset.lower()
NUM_OF_DATA = args.num_of_data
MODEL_PATH = f'/path/to/model/{args.model_name}'
FORMAT = ''

test_suffix =''
test_suffix = args.setting
if args.is_test == "False":
    is_test=False
else:
    is_test=True

if args.model_config:
    with open(args.model_config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    reflect_llm = make_generator(config["reflect_model_config"])
    if config["run_config"]["single_model"]:
        reason_llm = reflect_llm
    else:
        reason_llm = make_generator(config["reason_model_config"])
        print("using device 0 to load reflect model")

else:
    reason_llm = VLLMGenerator(MODEL_PATH)
    reflect_llm = reason_llm
sys.path.append("..")
root = "../root/"

few_shots = ""
data = []

print(f"args.reflection: {args.reflection}")
use_first_answer = args.use_first_answer
print(f"use_first_answer: {use_first_answer}")

if args.existing_dataset is None:
    data = get_dataset(DATASET, NUM_OF_DATA, is_test=is_test)
    use_first_answer=False
else:
    with open(args.existing_dataset, 'r', encoding='utf-8') as file:
        ii = 0
        for line in file:
            if NUM_OF_DATA!=0 and i>=NUM_OF_DATA: break
            i+=1
            item = json.loads(line)
            data.append(item)



FORMAT = ''
match DATASET:
    case "hotpot": fewshots, FORMAT = HOTPOT_FEWSHOTS, HOTPOT_FORMAT
    case "logiqa": fewshots, FORMAT = LOGIQA_FEWSHOTS, LOGIQA_FORMAT
    case "math": fewshots, FORMAT = MATH_FEWSHOTS, MATH_FORMAT
    case "mbpp": fewshots, FORMAT = MBPP_FEWSHOTS, MBPP_FORMAT
    case "bigbench": fewshots, FORMAT = BIGBENCH_FEWSHOTS, BIGBENCH_FORMAT
    case "bigbenchfree": fewshots, FORMAT = BIGBENCH_FREE_FEWSHOTS, BIGBENCH_FREE_FORMAT
    case _: raise ValueError("Invalid dataset")

if DATASET == "bigbenchfree":
    is_free_text = True
    print("is_free_text: True")
else:
    is_free_text = False
demand_str = ""
demand_type = args.demand_type
print(f"demand_type: {demand_type}")
if demand_type>0 and demand_type<=32:
    demand_str = DEMAND_TYPES[INT_TO_DEMAND_TYPES[args.demand_type]]
else:
    print("Invalid demand type.")

log = ""
agents = []
SAMPLE_SIZE = 2

model_name = args.model_name

if args.output_file is None:
    if is_test:
        output_file = f"data_test/{model_name}/{DATASET}_{METHOD}-generated_{NUM_OF_DATA}_test_{test_suffix}.jsonl"
        SAMPLE_SIZE = 1
    else:
        output_file = f"data_{model_name}/{DATASET}_{METHOD}-generated_{NUM_OF_DATA}_{demand_type}_{args.reflection}_{args.use_scratchpad}.jsonl"
else:
    output_file = args.output_file
existing_results = []
print(output_file)
try:
    with open(output_file, "r") as f:
        for line in f:
            existing_results.append(json.loads(line))
except FileNotFoundError:
    pass


existing_ids = {result["id"] for result in existing_results}

def process_row(row):
    s_t=time.time()
    if row["id"] in existing_ids:
        return None

    if 'test_list' in row:
        data = {
            "question": row["question"],
            "test_list": row["test_list"],
            "answer": row["answer"],
            "id": row["id"],
        }
    else:
        data = {
            "question": row["question"],
            "answer": row["answer"],
            "id": row["id"],
        }

    is_react = True
    if METHOD == "COT":
        is_react = False
    if DATASET == "mbpp":
        env = MBPPEnv(
            ground_truth=row["answer"],
            test_list=row["test_list"],
            is_react=is_react,
        )
    elif DATASET == "math":
        env = MATHEnv(
            ground_truth=row["answer"],
            is_react=is_react,
        )
    elif DATASET == "bigbench":
        env = BigbenchEnv(
            ground_truth=row["answer"],
            is_react=is_react,
        )
    elif DATASET == "bigbenchfree":
        env = BigbenchfreeEnv(
            ground_truth=row["answer"],
            is_react=is_react,
        )
    else:
        env = LogiQAEnv(
            ground_truth=row["answer"],
            is_react=is_react,
        )
    reason_prompt = REASON_PROMPT.replace('{format}', FORMAT)
    match METHOD:
        case "COT":
            if is_test:
                agent = BatchCOTReflectAgent(
                    question=row["question"],
                    answer=row["answer"],
                    reason_llm=reason_llm,
                    reflect_llm=reflect_llm,
                    env=env,
                    agent_prompt=reason_prompt,
                    reflect_prompt=REFLECTION_PROMPT_TEST,
                    examples=fewshots,
                    demand=demand_str
                )
            else:
                agent = BatchCOTReflectAgent(
                    question=row["question"],
                    answer=row["answer"],
                    reason_llm=reason_llm,
                    reflect_llm=reflect_llm,
                    env=env,
                    agent_prompt=reason_prompt,
                    reflect_prompt=REFLECTION_PROMPT,
                    examples=fewshots,
                    demand=demand_str
                )
        case "ReAct":
            if is_test:
                agent = BatchReactReflectAgent(
                    question=row["question"],
                    answer=row["answer"],
                    reason_llm=reason_llm,
                    reflect_llm=reflect_llm,
                    env=env,
                    agent_prompt=HOTPOTQA_REACT_PROMPT,
                    reflect_prompt=REFLECTION_PROMPT_TEST,
                    examples=HOTPOTQA_REACT_EXAMPLES,
                    demand=demand_str
                )
            else:
                agent = BatchReactReflectAgent(
                    question=row["question"],
                    answer=row["answer"],
                    reason_llm=reason_llm,
                    reflect_llm=reflect_llm,
                    env=env,
                    agent_prompt=HOTPOTQA_REACT_PROMPT,
                    reflect_prompt=REFLECTION_PROMPT,
                    examples=HOTPOTQA_REACT_EXAMPLES,
                    demand=demand_str
                )
        case _:
            raise ValueError(f"Invalid method: {METHOD}")
    agents.append(agent)
    s_t=time.time()
    output = []
    if use_first_answer == True:
        print("using first answer")
        if "output" in row:
            curr_output = row['output'][0]
            agent.generated_answer = curr_output['generated_answer']
            agent.reflections = []
            agent.scratchpad = curr_output['scratchpad']
        else:
            agent.reflections = []
            agent.scratchpad = row["first_reason"]
            agent.generated_answer = row["first_answer"]
    else:
        if(args.use_scratchpad.lower() == "true"):
            print("use scratchpad")
            agent.run(trail=1, setting=2,is_free_text=is_free_text)
        else:
            print("not use scratchpad")
            agent.run(trail=1, setting=1,is_free_text=is_free_text)
    is_correct = agent.is_correct()
    error = None
    if isinstance(is_correct, tuple):
        error = is_correct[1]
        is_correct = is_correct[0]
    output.append(
        {
            "generated_answer": agent.generated_answer,
            "reflections": agent.reflections,
            "scratchpad": agent.scratchpad,
            "is_correct": is_correct,
            "error": error,
            "trail": 1,
            "reasoning_source": reason_llm.model_id,
            "reflection_source": reflect_llm.model_id,
        }
    )
    s_t=time.time()
    if is_correct:
        print(f"Answer t=1: {agent.generated_answer}")
    elif args.reflection=="1":
        reflections = agent.prompt_reflection(sample_size=SAMPLE_SIZE)
        s_t=time.time()
        if not isinstance(reflections, list):
            reflections = [reflections]
        for i, r in enumerate(reflections):
            if(args.use_scratchpad.lower() == "true"):
                print("use scratchpad")
                agent.run(trail=2,setting=2,reflection=r,is_free_text=is_free_text)
            else:
                print("not use scratchpad")
                agent.run(trail=2,setting=1,reflection=r,is_free_text=is_free_text)
            s_t=time.time()
            is_correct = agent.is_correct()
            error = None
            if isinstance(is_correct, tuple):
                error = is_correct[1]
                is_correct = is_correct[0]
            print(f"Answer t=2-{i}: {agent.generated_answer}")
            output.append(
                {
                    "generated_answer": agent.generated_answer,
                    "reflections": agent.reflections,
                    "scratchpad": agent.scratchpad,
                    "is_correct": is_correct,
                    "error": error,
                    "trail": 2,
                    "reasoning_source": reason_llm.model_id,
                    "reflection_source": reflect_llm.model_id,
                }
            )
    elif args.reflection == "2":
        print("use reflection")
        if "output" in row and isinstance(row["output"], list):
            outputs = row["output"]
            reflections = []
            for out in outputs:

                if "reflections" in out:
                    reflection = out["reflections"]
                    if isinstance(reflection, str):
                        reflections.append(reflection)

        s_t=time.time()
        if not isinstance(reflections, list):
            reflections = [reflections]
        for i, r in enumerate(reflections):
            if(args.use_scratchpad.lower() == "true"):
                print("use scratchpad")
                agent.run(trail=2,setting=2,reflection=r,is_free_text=is_free_text)
            else:
                print("not use scratchpad")
                agent.run(trail=2,setting=1,reflection=r,is_free_text=is_free_text)
            s_t=time.time()
            is_correct = agent.is_correct()
            error = None
            if isinstance(is_correct, tuple):
                error = is_correct[1]
                is_correct = is_correct[0]
            print(f"Answer t=2-{i}: {agent.generated_answer}")
            output.append(
                {
                    "generated_answer": agent.generated_answer,
                    "reflections": agent.reflections,
                    "scratchpad": agent.scratchpad,
                    "is_correct": is_correct,
                    "error": error,
                    "trail": 2,
                    "reasoning_source": reason_llm.model_id,
                    "reflection_source": "pre-stored",
                }
            )
    elif args.reflection == "3":
        print("without reflection")
        s_t=time.time()
        if(args.use_scratchpad.lower() == "true"):
            print("use scratchpad")
            agent.run(trail=2,setting=2,is_free_text=is_free_text)
        else:
            print("not use scratchpad")
            agent.run(trail=2,setting=1,is_free_text=is_free_text)
        is_correct = agent.is_correct()
        error = None
        if isinstance(is_correct, tuple):
            error = is_correct[1]
            is_correct = is_correct[0]
        print(f"Answer t=2: {agent.generated_answer}")
        output.append(
            {
                "generated_answer": agent.generated_answer,
                "reflections": agent.reflections,
                "scratchpad": agent.scratchpad,
                "is_correct": is_correct,
                "error": error,
                "trail": 2,
                "reasoning_source": reason_llm.model_id,
                "reflection_source": "none",
            }
        )
    data["output"] = output
    return data

start_time = time.time()
with open(output_file, "a") as f:
    start_time = time.time()
    for i, row in enumerate(data):
        if use_second_answer and len(row["output"]) > 1:
            result = row
        else:
            result = process_row(row)
        if result is not None:
            f.write(json.dumps(result) + "\n")
print('finish')
