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
    """
    加载指定的数据集，并根据需要随机选取一定数量的样本。

    :param dataset_name: 数据集名称，如 "LogiQA" 或 "MATH"
    :param number: 需要选取的样本数量，0 表示不限制
    :param seed: 随机种子，确保选取的样本可复现
    :return: 处理后的数据列表，每项包含 "id", "question" 和 "answer"
    """
    output = []
    random.seed(seed)  # 固定随机种子
    dataset_name = dataset_name.lower()
    if dataset_name == "logiqa":
        # 读取本地的 LogiQA 数据集
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
        # 加载 Hugging Face 的 MATH 数据集
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
                # 构建 question 和 answer
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

    # 如果需要随机选取样本
    if number > 0:
        if number > len(output):
            print(f"警告：请求的样本数量 {number} 超过了数据集的总样本数 {len(output)}，将返回所有样本。")
            number = len(output)
        output = random.sample(output, number)  # 随机选取指定数量的样本

    if train_eval == 'eval':
        # 如果是评估模式，返回最多200条数据，且不超过output的10%
        eval_size = min(200, int(len(output) * 0.1))  # 计算eval数据量
        return output[:eval_size]  # 返回前eval_size条数据
    elif train_eval == 'train':
        # 如果不是评估模式，返回除了eval数据之外的其他数据
        eval_size = min(200, int(len(output) * 0.1))  # 计算eval数据量
        return output[eval_size:]  # 返回eval_size之后的数据
    else:
        return output

print(max(len(item['answer'].split()) for item in get_dataset('bigbenchfree') if item.get('answer') is not None))
