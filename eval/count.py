import json
import sys

def process_jsonl(file_path):

    first_output_correct = 0
    second_output_correct = 0
    total_records = 0


    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            total_records += 1


            outputs = data['output']
            trail1_correct = outputs[0]['is_correct']
            trail2_correct = outputs[1]['is_correct'] if len(outputs) > 1 else False

            if trail1_correct:
                first_output_correct += 1
            else:
                if trail2_correct:
                    second_output_correct += 1

    # count correctness
    acc_t1 = (first_output_correct / total_records * 100) if total_records > 0 else 0
    acc_t2 = ((first_output_correct + second_output_correct) / total_records * 100) if total_records > 0 else 0

    # print correctness
    print(f"first_output_correct: {first_output_correct}")
    print(f"acc@t1: {acc_t1:.2f}%")
    print(f"second_output_correct: {second_output_correct}")
    print(f"acc@t2: {acc_t2:.2f}%")


if __name__ == "__main__":
    process_jsonl(sys.argv[1])
