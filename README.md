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
- [Training Guide](#training-guide)
- [Evaluation](#evaluation)
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
You can download the **ReflectEvo-460K** data here: 

You can also access our sample data [data/](data/).


For Reflection Generation, run
```
python run/run.py --method COT --dataset Logiqa --demand_type 1 --model_name /path/to/model
```


## Training Guide

For full-parameter SFT, use
```
```

For parameter-efficient fine-tuning (PEFT), use
```
python train_c2.py --task logiqa --input_data /path/to/training/data --output /path/to/output --model_path /path/to/model
```

For PPO, use
```
```

## Evaluation
For evaluation, use the following command to test the performance of the model for both one stage training and DPO training:

```
python run/run.py --method COT --dataset Logiqa --num_of_data 0 -is_test True  --model_name Meta-Llama-3-8B-Instruct
```

Use the following command to test the performance of the model for two stage training:

```
python run_c2/run.py --method COT --dataset Logiqa --num_of_data 0 -is_test True  --model_name Meta-Llama-3-8B-Instruct
```

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
python eval/count.py "your_result_path"
```
