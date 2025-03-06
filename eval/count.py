import json
from collections import defaultdict
import sys

def process_jsonl(file_path):
    # 初始化统计数据
    single_output_correct = 0
    single_output_incorrect = 0
    any_output_correct = 0
    multiple_output_highest_correct_all = 0
    multiple_output_correct_higher_score_all = 0
    multiple_output_none_correct = 0
    total_records = 0

    # 用于存储不同类型的分数列表
    scores_distribution = {
        "single_output_correct": [],
        "single_output_incorrect": [],
        "any_output_correct": [],
        "multiple_output_highest_correct_all": [],
        "multiple_output_correct_higher_score_all": [],
        "multiple_output_none_correct": []
    }
    correct_at_first=[] 
    correct_at_second=[]
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            total_records += 1
            outputs = data['output']
            scores = [int(output.get('score', 0)) for output in outputs]

            correct_outputs = [output for output in outputs if output['is_correct']]
            incorrect_outputs = [output for output in outputs if not output['is_correct']]
            #if correct_outputs:
                #print(data['id'])

            if len(outputs) == 1:
                if outputs[0]['is_correct']:
                    single_output_correct += 1
                    correct_at_first.append(data["id"])
                    scores_distribution["single_output_correct"].extend(scores)
                else:
                    single_output_incorrect += 1
                    scores_distribution["single_output_incorrect"].extend(scores)
            else:
                if correct_outputs:
                    if not outputs[0]['is_correct']:
                        correct_at_second.append(data['id'])
                    any_output_correct += 1
                    scores_distribution["any_output_correct"].extend(scores)

                    highest_score = max(scores)
                    highest_outputs = [output for output in outputs if int(output.get('score', 0)) == highest_score]
                    if all(output['is_correct'] for output in highest_outputs):
                        multiple_output_highest_correct_all += 1
                        scores_distribution["multiple_output_highest_correct_all"].extend(scores)

                    correct_scores = [int(output.get('score', 0)) for output in correct_outputs]
                    incorrect_scores = [int(output.get('score', 0)) for output in incorrect_outputs]
                    if correct_scores and incorrect_scores and min(correct_scores) > max(incorrect_scores):
                        multiple_output_correct_higher_score_all += 1
                        scores_distribution["multiple_output_correct_higher_score_all"].extend(scores)
                else:
                    multiple_output_none_correct += 1
                    scores_distribution["multiple_output_none_correct"].extend(scores)

    # 计算比例并转换为百分比，保留两位小数
    single_output_correct_ratio = round((single_output_correct / total_records) * 100, 2) if total_records > 0 else 0
    single_output_incorrect_ratio = round((single_output_incorrect / total_records) * 100, 2) if total_records > 0 else 0
    any_output_correct_ratio = round((any_output_correct / total_records) * 100, 2) if total_records > 0 else 0

    multiple_output_highest_correct_ratio_total = round((multiple_output_highest_correct_all / total_records) * 100, 2) if total_records > 0 else 0
    multiple_output_highest_correct_ratio_any = round((multiple_output_highest_correct_all / any_output_correct) * 100, 2) if any_output_correct > 0 else 0

    multiple_output_correct_higher_score_ratio_total = round((multiple_output_correct_higher_score_all / total_records) * 100, 2) if total_records > 0 else 0
    multiple_output_correct_higher_score_ratio_any = round((multiple_output_correct_higher_score_all / any_output_correct) * 100, 2) if any_output_correct > 0 else 0

    multiple_output_none_correct_ratio = round((multiple_output_none_correct / total_records) * 100, 2) if total_records > 0 else 0

    # 计算各类分数分布
    score_distribution_result = {}
    for key, score_list in scores_distribution.items():
        score_distribution_result[key] = {score: round((score_list.count(score) / len(score_list)) * 100, 2) for score in set(score_list)}

    # 打印结果
    print(f"0. 第一次答错: {total_records-single_output_correct}")
    print(f"0. Reflect后答对: {any_output_correct}")
    print(f"0. 总数据: {total_records}")
    print(f"1. 只有一个output就is_correct的比例: {single_output_correct_ratio}%")
    print(f"2. 只有一个output不是is_correct的比例: {single_output_incorrect_ratio}%")
    print(f"3. 任何一个output是is_correct的比例: {any_output_correct_ratio}%")
    print(f"4. 多个output，有reflection的之中，score最高的is_correct，且满足3的比例（在全体中的百分比）: {multiple_output_highest_correct_ratio_total}%")
    print(f"   多个output，有reflection的之中，score最高的is_correct，且满足3的比例（在3中的百分比）: {multiple_output_highest_correct_ratio_any}%")
    print(f"5. 多个output，有reflection的之中，is_correct的score比并非is_correct的高，且满足3的比例（在全体中的百分比）: {multiple_output_correct_higher_score_ratio_total}%")
    print(f"   多个output，有reflection的之中，is_correct的score比并非is_correct的高，且满足3的比例（在3中的百分比）: {multiple_output_correct_higher_score_ratio_any}%")
    print(f"6. 多个output，但是没有is_correct的比例: {multiple_output_none_correct_ratio}%")
    print(f"7. 各类score分布: {score_distribution_result}")
    print(f"8. 第一次就回答对的id: {correct_at_first}")
    print(f"8. refl后回答对的id: {correct_at_second}")

# 调用函数并传入JSONL文件路径
process_jsonl(sys.argv[1])
