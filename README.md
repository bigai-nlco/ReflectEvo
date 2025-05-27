<h1 align="center">ReflectEvo: Improving Meta Introspection of Small LLMs  by Learning Self-Reflection</h1>

  
<p align="center">
    <a href="https://huggingface.co/datasets/bigai-nlco/ReflectionEvo">
        <img alt="Documentation" src="https://img.shields.io/badge/Dataset-HF Data-yellow.svg">
    </a>
    <a href="https://arxiv.org/abs/2505.16475">
        <img alt="Documentation" src="https://img.shields.io/badge/Paper-arXiv-red.svg">
    </a>
</p>

We propose a novel pipeline **ReflectEvo**, to automatically generate self-reflection data and leverage self-training to enhance LLM’s reflection capability. Building on this pipeline, we curate a large scale, diverse, and unsupervised reflection learning dataset **ReflectEvo-460k** containing 460k reflection samples derived from 17 source datasets spanning 10 tasks and domains.

![Overall Pipeline](assets/overall.png)

*Overview pipeline of ReflectEvo.* There are four key stages: (1) Initial thoughts and answers are collected from 10 tasks and 17 datasets, (2) Reflection Generation for erroneous samples including self-reflection and self-correction, (3) Reflection curation (4) Reflection Tuning to enhance LLMs via self-training.

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
   pip install -r requirements.txt
   ```
   
## Reflection Generation
### Download the Data
You can download the **ReflectEvo-460K** data here: https://disk.pku.edu.cn/link/AADBE037EB31AB48878545A552DE2C9ACC

You can also access our sample data here [data/](data/).

### Generate Reflection
For Reflection Generation, run
```
python -m run.run --dataset Logiqa --demand_type 1 --model_name /path/to/model
```
Open-source models can be downloaded and loaded from [Models/](Models/) by default, you can change the path via `--model_name`

Datasets can be selected via `--dataset`. Options include LogiQA, MATH, MBPP, Bigbench, and Bigbenchfree (a filtered subset of free-text tasks from BIG-bench).

Instructions are specified using `--demand_type`, which corresponds to an instruction ID from the instruction pool (see Appendix B.1 for details).

Reflection behavior is controlled with `--reflection`. It defaults to 1 (regenerate). You can also use 2 (load stored reflection) or 3 (skip reflection).


## Training Guide

For two stage training with D<sup>+</sup>, we use full-parameter supervised fine-tuning (SFT)(See Appendix B.2). First use
```
PYTHONPATH=. torchrun --master-port 5508 --nproc_per_node=1 train/train_SFT_two_stage_1.py --task logiqa --num_epochs 3 --resume False --output /your/output/model/name --model_path /your/model/path ---ebs 20 --bs 8 --ss steps --wd 0.01 --lr 1e-3 --gas 4
```

then use
```
PYTHONPATH=. torchrun --master-port 5507 --nproc_per_node=1 train/train_SFT_two_stage_2.py --task logiqa --num_epochs 5 --resume False --output /your/output/model/name --model_path /your/model/path --ss steps --ebs 50 --bs 8 --wd 0.01 --lr 1e-3 --gas 4 --folder /your/train/data/path
```

Use `--task` to specify the dataset.

Use `--model_path` to provide the path to the base model.

Use `--folder` to specify the folder containing the training data.

Use `--output` to save results or logs.


For one stage training with D<sup>+</sup>, we use Low-Rank Adaptation (LoRA)-based Parameter-Efficient Fine-Tuning (PEFT)(See Appendix B.2). Use
```
PYTHONPATH=. python train/train_SFT_one_stage.py --task logiqa --input_data /path/to/training/data --output /path/to/output --model_path /path/to/model
```

Use `--input_data` to specify the folder containing the training data.


For Direct Preference Optimization(DPO) training with both D<sup>±</sup> and D<sup>pref</sup>(See Appendix B.2), use
```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/deepspeed_zero3.yaml --num_processes=4 run_dpo.py configs/DPO_train_config.yaml
```
Use DeepSpeed Zero3 to accelerate and run run_dpo.py. The num_processes parameter represents the number of GPUs, and DPO_train_config.yaml contains the training configuration.

## Evaluation
### Generate Results

For evaluation, use the following command to test the performance of the model for both SFT one stage training and DPO training:

```
python -m run.run --dataset Logiqa --is_test True  --model_name /path/to/model --model_config /path/to/model/config
```


Use the following command to test the performance of the model for SFT two stage training:

```
python -m run.run_PEFT --dataset Logiqa --is_test True  --model_name /path/to/model --model_config /path/to/model/config
```

#### Arguments

Set `--is_test` to "True" for evaluation.

Use `--model_config` to specify the path to a YAML file containing the model configuration.

### Evaluate Performance

For datasets with multiple-answer questions, use the following command to evaluate the model's performance:

```
python -m eval.count "path/to/your/results"
```

For questions with free-text answers, use the following command to evaluate the model's performance:

```
python -m eval.count_f1 "path/to/your/results"
```

## Citation

## Acknowledgment

