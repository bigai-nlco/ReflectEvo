import json
import re
import argparse

# extract f1 score
def extract_last_f1_score(scratchpad):
    matches = re.findall(r'F1 Score:\s*([\d\.]+)', scratchpad)
    if matches:
        return float(matches[-1])
    return None

def process_file(input_file):
    total_samples = 0
    trail1_correct_count = 0
    improved_count = 0
    improved_indices = []

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line_index, line in enumerate(infile, start=1):
            data = json.loads(line)
            output_list = data.get("output", [])

            if len(output_list) >= 1:
                total_samples += 1

                trail1 = output_list[0]
                if trail1.get("is_correct", False) is True:
                    trail1_correct_count += 1

                if len(output_list) >= 2:
                    trail2 = output_list[1]

                    f1_score1 = trail1.get("F1_score", None)
                    f1_score2 = trail2.get("F1_score", None)

                    if f1_score1 is None:
                        f1_score1 = extract_last_f1_score(trail1.get("scratchpad", "")) or 0.0
                    if f1_score2 is None:
                        f1_score2 = extract_last_f1_score(trail2.get("scratchpad", "")) or 0.0

                    if f1_score2 > f1_score1:
                        improved_count += 1
                        improved_indices.append(line_index)

    # count score
    trail1_correct_ratio = (trail1_correct_count / 500) * 100
    improved_ratio = (improved_count / 500) * 100

    # print score
    print(f"total samples: {total_samples}")
    print(f"trail 1 correct count: {trail1_correct_count}")
    print(f"trail 1 correctness: {trail1_correct_ratio:.2f}%")
    print(f"improved count: {improved_count}")
    print(f"improved ratio: {improved_ratio:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL file to analyze F1 score improvements.")
    parser.add_argument("input_file", type=str, help="Path to the input JSONL file")
    args = parser.parse_args()

    process_file(args.input_file)
