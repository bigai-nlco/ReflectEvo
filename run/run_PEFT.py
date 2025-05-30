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
    REFLECTION_PROMPT_TEST_C2,
    DEMAND_TYPES
)
from prompts.fewshots import (
    HOTPOTQA_FEWSHOTS,
    LOGIQA_FEWSHOTS,
    MATH_FEWSHOTS,
    MBPP_FEWSHOTS,
    BIGBENCH_FEWSHOTS,
    BIGBENCH_FREE_FEWSHOTS
)
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
parser.add_argument("--dataset", "-d", type=str, help="LogiQA, MATH or MBPP")
parser.add_argument("--existing_dataset", "-ed", type=str, help="Regenerate it with the data generated earlier. Fill in the file path here")
parser.add_argument("--use_first_answer", "-u", type=str, help="Whether to keep the first answer in existing dataset", default="true")
parser.add_argument("--num_of_data", "-n", type=int, help="number of data, 0 for all data", default=0)
parser.add_argument("--demand_type", "-dt", type=str, help="The type of demand for the reflection task. Choose from 1 to 11.", default=1)
parser.add_argument("--output_file", "-o", type=str, help="output directory", required=False)
parser.add_argument("--model_name", "-mn", type=str, help="model name", default='Meta-Llama-3-8B-Instruct')
parser.add_argument("--reflection", "-r", type=str, help="Whether to use refl", default="True")
parser.add_argument("--model_config", "-mc", type=str, help="YAML file address for model configuration.", default="")
parser.add_argument("--is_test", "-t", type=str, default="False")
parser.add_argument("--setting", "-s", type=str, default="")
parser.add_argument("--device1", "-d1", type=int, default=-1)
parser.add_argument("--device2", "-d2", type=int, default=-1)
args = parser.parse_args()
args.reflection = args.reflection.lower() == "true"
args.use_first_answer = args.use_first_answer.lower() == "true"
multi_device = False
d1 = args.device1
d2 = args.device2
if args.device1 != args.device2:
    multi_device = True
METHOD = args.method
if args.dataset:
    DATASET = args.dataset.lower()
NUM_OF_DATA = args.num_of_data
MODEL_PATH = f'/path/to/model/{args.model_name}'
FORMAT = ''
test_suffix =''
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
else:
    reason_llm = VLLMGenerator(MODEL_PATH)
    reflect_llm = reason_llm
test_suffix = args.setting
print("reflect_llm path: ", reflect_llm.model_id)

sys.path.append("..")
root = "../root/"

few_shots = ""
data = []

print(args.reflection)

use_first_answer = args.use_first_answer
print(use_first_answer)
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
    print(f"dataset path: {args.existing_dataset}")



FORMAT = ''
match DATASET:
    case "hotpot": fewshots, FORMAT = HOTPOT_FEWSHOTS, HOTPOT_FORMAT
    case "logiqa": fewshots, FORMAT = LOGIQA_FEWSHOTS, LOGIQA_FORMAT
    case "math": fewshots, FORMAT = MATH_FEWSHOTS, MATH_FORMAT
    case "mbpp": fewshots, FORMAT = MBPP_FEWSHOTS, MBPP_FORMAT
    case "bigbench": fewshots, FORMAT = BIGBENCH_FEWSHOTS, BIGBENCH_FORMAT
    case "bigbenchfree": fewshots, FORMAT = BIGBENCH_FREE_FEWSHOTS, BIGBENCH_FREE_FORMAT
    case _: raise ValueError("Invalid dataset")

demand_str = ""
demand_type = args.demand_type
print(f"demand_type: {demand_type}")

if demand_type in DEMAND_TYPES:
    demand_str = DEMAND_TYPES[demand_type]
else:
    print(f"Invalid demand type: {demand_type}. Please provide a valid key from DEMAND_TYPES.")
    exit(1)

log = ""
agents = []
SAMPLE_SIZE = 2

model_name = args.model_name

if args.output_file is None:
    if is_test:
        output_file = f"data/data_test/{model_name}/{DATASET}_{METHOD}-generated_{NUM_OF_DATA}_test_{test_suffix}.jsonl"
        SAMPLE_SIZE = 1
    else:
        output_file = f"data/data_raw/{model_name}/{DATASET}_{METHOD}-generated_{NUM_OF_DATA}_{demand_type}.jsonl"
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
    elif DATASET == "logiqa":
        env = LogiQAEnv(
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
        raise ValueError("no match dataset.")
    reason_prompt = REASON_PROMPT.replace('{format}', FORMAT)
    match METHOD:
        case "COT":
            if is_test:
                if test_suffix == 'c2' or test_suffix == 'c2_qs' or test_suffix[:2]=='c2':
                    agent = BatchCOTReflectAgent(
                        question=row["question"],
                        answer=row["answer"],
                        reason_llm=reason_llm,
                        reflect_llm=reflect_llm,
                        env=env,
                        agent_prompt=reason_prompt,
                        reflect_prompt=REFLECTION_PROMPT_TEST_C2,
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
        case _:
            raise ValueError(f"Invalid method: {METHOD}")
    agents.append(agent)
    s_t=time.time()
    output = []
    if use_first_answer == True:
        print("use_first_answer")

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
        agent.run_c2(reflection='',trail=1)
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
            "reason_prompt": reason_prompt,
            "reflection_prompt": ""
        }
    )
    s_t=time.time()
    if is_correct:
        print(f"Answer t=1: {agent.generated_answer}")

    elif args.reflection:
        reflections = agent.prompt_reflection(sample_size=SAMPLE_SIZE)
        s_t=time.time()
        if not isinstance(reflections, list):
            reflections = [reflections]


        for i, r in enumerate(reflections):
            agent.run_c2(reflection=r,trail=2)
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
                    "reason_prompt": reason_prompt,
                    "reflection_prompt": agent.reflect_prompt
                }
            )

    data["output"] = output
    return data

start_time = time.time()
with open(output_file, "a") as f:
    start_time = time.time()
    for i, row in enumerate(data):
        result = process_row(row)
        if result is not None:
            f.write(json.dumps(result) + "\n")

