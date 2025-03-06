import json
import re
import argparse

# 提取有效的 F1_score
def extract_last_f1_score(scratchpad):
    matches = re.findall(r'F1 Score:\s*([\d\.]+)', scratchpad)
    if matches:
        return float(matches[-1])
    return None

def process_file(input_file):
    total_samples = 0  # 总数据量
    trail1_correct_count = 0  # 第一次正确的数量
    improved_count = 0  # trail2 F1 > trail1 F1 的数量
    improved_indices = []  # 存储 trail2 F1 > trail1 F1 的题目编号

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line_index, line in enumerate(infile, start=1):  # 使用 line_index 记录题目编号
            data = json.loads(line)
            output_list = data.get("output", [])

            # 样本计入总数据量
            if len(output_list) >= 1:
                total_samples += 1

                # 检查 trail1 是否正确
                trail1 = output_list[0]
                if trail1.get("is_correct", False) is True:
                    trail1_correct_count += 1

                # 检查是否有 trail2，统计改进数量
                if len(output_list) >= 2:
                    trail2 = output_list[1]

                    # 提取 trail1 和 trail2 的 F1_score
                    f1_score1 = trail1.get("F1_score", None)
                    f1_score2 = trail2.get("F1_score", None)

                    if f1_score1 is None:
                        f1_score1 = extract_last_f1_score(trail1.get("scratchpad", "")) or 0.0
                    if f1_score2 is None:
                        f1_score2 = extract_last_f1_score(trail2.get("scratchpad", "")) or 0.0

                    # 统计 trail2 的 F1 是否改进
                    if f1_score2 > f1_score1:
                        improved_count += 1
                        improved_indices.append(line_index)  # 记录编号

    # 计算比例
    trail1_correct_ratio = (trail1_correct_count / 500) * 100
    improved_ratio = (improved_count / 500) * 100

    # 打印结果
    print(f"总数据数量: {total_samples}")
    print(f"第一次正确的数量: {trail1_correct_count}")
    print(f"第一次正确的比例: {trail1_correct_ratio:.2f}%")
    print(f"trail2 F1 > trail1 F1 的数量: {improved_count}")
    print(f"trail2 F1 改进的比例: {improved_ratio:.2f}%")
    print("\ntrail2 F1 > trail1 F1 的题目编号:")
    for index in improved_indices:
        print(index)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL file to analyze F1 score improvements.")
    parser.add_argument("input_file", type=str, help="Path to the input JSONL file")
    args = parser.parse_args()

    process_file(args.input_file)
