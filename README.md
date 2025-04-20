<h1 align="center">Improving Meta Introspection of Small LLMs by Learning Self-Reflection from Self-Generated Data</h1>
<p align="center">

We propose a novel pipeline **ReflectEvo**, to automatically generate self-reflection data and leverage self-training to enhance LLM’s reflection capability. 

Building on this pipeline, we curate a large scale, diverse, and unsupervised reflection learning dataset **ReflectEvo-460k** containing 460k reflection samples derived from 17 source datasets spanning 10 tasks and domains.

![Overall Pipeline](assets/overall.png)

*Overview pipeline of ReflectEvo.* There are four key stages: (1) Initial thoughts and answers are collected from 10 tasks and 17 datasets, (2) Reflection Generation for erroneous samples including self-reflection and self-correction, (3) Reflection curation with positive, negative, and self-reflection samples, and (4) Reflection Tuning to enhance LLMs via self-training.

## 📖 Table of contents
- [Statistics of ReflectEvo-460K](#statistics-of-reflectevo-460k)
- [Installation](#installation)
- [Reflection Generation](#reflection-generation)
  - [Download the Data](#download-the-data)
  - [Generate Reflection](#generate-reflection)
- [Training Guide](#training-guide)
- [Evaluation](#evaluation)
  - [Generate Results](#generate-results)
  - [Evaluate Performance](#evaluate-performance)
- [Citation](#citation)
- [Acknowledgment](#acknowledgment)
  
## 📌Statistics of ReflectEvo-460K
![Statistics](assets/statistics.png)
## Installation
Clone this repository and navigate to ReflectEvo folder
   
   ```
   git clone https://github.com/Sheng-Shu/ReflectEvo.git
   cd ReflectEvo
   ```
   
Install Package
   
   ```
   pip install -r requirements.txt
   ```
   
## Reflection Generation
### Download the Data
You can download the **ReflectEvo-460K** data here: https://disk.pku.edu.cn/link/AADBE037EB31AB48878545A552DE2C9ACC

You can also access our sample data here [data/](data/).

### Generate Reflection
For Reflection Generation, run
```
python run.py --dataset Logiqa --demand_type 1 --model_name /path/to/model
```
#### Arguments
--dataset (required): The dataset to use. Options include:
  LogiQA, MATH, MBPP, Bigbench, Bigbenchfree(Filtered subset of freetext tasks from BIG-bench).

--demand_type (required): Instruction ID from the instruction pool. (See Appendix B.1).

--model_name (required): Name of the model to use.

--reflection (optional): How to use reflection in the task. Options: (Default: 1)
  1: regenerate reflection
  2: use pre-stored reflection
  3: skip reflection generation

--is_test(optional): Use "False" for data generation.

--use_scratchpad (required): Whether to include the reasoning process from the first round in the second-round prompt.

--existing_dataset (optional): Path to a previously generated dataset. If provided, the script will load and reuse this dataset.

--use_first_answer (optional): Whether to keep the first answer from the existing dataset.

--use_second_answer (optional): Whether to keep the second answer from the existing dataset.

--num_of_data (optional): Number of samples to process. Set to 0 to process all available data.

--output_file (optional): Output file path or directory for results.

--model_config (optional): Path to YAML file specifying model configuration.

--setting (optional):
Custom identifier or tag for this run.

## Training Guide

For full-parameter SFT, first use
```
torchrun --master-port 5508 --nproc_per_node=1 train_SFT_two_stage_1.py --version 1 --task logiqa --num_epochs 3 --resume False --output /your/output/model/name --model_path /your/model/path --template 1 --ebs 20 --bs 8 --ss steps --wd 0.01 --lr 1e-3 --gas 4
```

then use
```
torchrun --master-port 5507 --nproc_per_node=1 train_SFT_two_stage_2.py --version 1 --task logiqa --num_epochs 5 --resume False --output /your/output/model/name --model_path /your/model/path --template 1 --ss steps --ebs 50 --bs 8 --wd 0.01 --lr 1e-3 --gas 4 --folder /your/train/data/path
```

For parameter-efficient fine-tuning (PEFT), use
```
python run_SFT_one_stage.py --task logiqa --input_data /path/to/training/data --output /path/to/output --model_path /path/to/model
```

For DPO, use
```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/deepspeed_zero3.yaml --num_processes=4 run_dpo.py configs/DPO_train_config.yaml
```

## Evaluation
### Generate Results

For evaluation, use the following command to test the performance of the model for both SFT one stage training and DPO training:

```
python run.py --dataset Logiqa --is_test True  --model_name /path/to/model --model_config /path/to/model/config
```

Use the following command to test the performance of the model for SFT two stage training:

```
python run_PEFT.py --dataset Logiqa --is_test True  --model_name /path/to/model --model_config /path/to/model/config
```


### Evaluate Performance

For datasets with multiple-answer questions, use the following command to evaluate the model's performance:

```
python eval/count.py "path/to/your/results"
```

For questions with free-text answers, use the following command to evaluate the model's performance:

```
python eval/count_f1.py "path/to/your/results"
```

## Citation

## Acknowledgment

