import json
import random
from datasets import load_dataset

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def get_dataset(dataset_name, number=0, seed=22, is_test=False, train_eval='', is_sliced=False, begin=0, end=0):
    output = []
    random.seed(seed)
    dataset_name = dataset_name.lower()
    if dataset_name == "logiqa":

        if is_test == True:
            file_path = "/path/to/logiqa_test.jsonl"
        else:
            file_path = "/path/to/LogiQA/Train.jsonl"
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                output.append({
                    "id": idx + 1,
                    "question": data["question"],
                    "answer": data["answer"]
                })
    elif dataset_name == "mbpp":
        if is_test == True:
            file_path = "/path/to/mbpp_test.jsonl"
        else:
            file_path = "/path/to/MBPP/Train.jsonl"
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                output.append({
                    "id": data["task_id"],
                    "question": 'You are an expert Python programmer, and here is your task: ' +  data["text"] + 'Your code should pass these tests: ' + str(data["test_list"]),
                    "answer": data["code"],
                    "test_list": data["test_list"]
                })
    elif dataset_name == "math":

        if is_test == True:
            file_path = "/path/to/test_data/math_test.jsonl"
            with open(file_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    data = json.loads(line)
                    solution = data["solution"]
                    answer = remove_boxed(last_boxed_only_string(solution))
                    output.append({
                        "id": idx + 1,
                        "question": data["problem"],
                        "solution": solution,
                        "answer": answer
                    })

        else:
            dataset = load_dataset("lighteval/MATH", split="train")
            for idx, item in enumerate(dataset):

                question = item["problem"]
                solution = item["solution"]
                answer = remove_boxed(last_boxed_only_string(solution))
                output.append({
                    "id": idx + 1,
                    "question": question,
                    "solution": solution,
                    "answer": answer
                })
    elif dataset_name == "bigbench":
        if is_test == True:
            with open("/path/to/bigbench_test.jsonl",'r',encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    data = json.loads(line)
                    output.append({
                        "id": idx + 1,
                        "question": data["question"],
                        "answer": data["answer"]
                    })
        else:
            with open("/path/to/bigbench_multiple_choice_one_answer.jsonl", 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    data = json.loads(line)
                    output.append({
                        "id": idx + 1,
                        "question": data["question"],
                        "answer": data["answer"]
                    })
    elif dataset_name == "bigbenchfree":
         if is_test == True:
            with open("/path/to/bigbenchfree_test.jsonl",'r',encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    data = json.loads(line)
                    output.append({
                        "id": idx + 1,
                        "question": data["question"],
                        "answer": data["answer"]
                    })
         else:
            with open("/path/to/bigbench_freetext.jsonl", 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    data = json.loads(line)
                    output.append({
                        "id": idx + 1,
                         "question": data["question"],
                        "answer": data["answer"]
                })
    if is_sliced:
        return output[begin:end]


    if number > 0:
        if number > len(output):
            number = len(output)
        output = random.sample(output, number)

    if train_eval == 'eval':

        eval_size = min(200, int(len(output) * 0.1))
        return output[:eval_size]
    elif train_eval == 'train':

        eval_size = min(200, int(len(output) * 0.1))
        return output[eval_size:]
    else:
        return output


