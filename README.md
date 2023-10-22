# Exploring the Potential of Large Language Models (LLMs) in Learning on Graphs


**UPDATE**: The pt file for Citeseer has some problems. Please use the latest version instead of the version inside small_data.zip

This is the official code repository for our paper [Exploring the Potential of Large Language Models (LLMs) in Learning on Graphs](https://arxiv.org/abs/2307.03393)



## Introduction
Learning on Graphs has attracted immense attention due to its wide real-world applications. The most popular pipeline for learning on graphs with textual node attributes primarily relies on Graph Neural Networks (GNNs), and utilizes shallow text embedding as initial node representations, which has limitations in general knowledge and profound semantic understanding. In recent years, Large Language Models (LLMs) have been proven to possess extensive common knowledge and powerful semantic comprehension abilities that have revolutionized existing workflows to handle text data. In this paper, we aim to explore the potential of LLMs in graph machine learning, especially the node classification task, and investigate two possible pipelines: LLMs-as-Enhancers and LLMs-as-Predictors. The former leverages LLMs to enhance nodes' text attributes with their massive knowledge and then generate predictions through GNNs. The latter attempts to directly employ LLMs as standalone predictors. We conduct comprehensive and systematical studies on these two pipelines under various settings. From comprehensive empirical results, we make original observations and find new insights that open new possibilities and suggest promising directions to leverage LLMs for learning on graphs. 

We provide the implementation of the following pipelines. 
### LLMs-as-Predictors
Check `ego_graph.py`, and directly use `ChatGPT` to do zero-shot/few-shot predictions. 
![LLMs-as-Predictors](https://github.com/CurryTang/Graph-LLM/blob/master/imgs/llm_as_predictor.png)

### LLMs-as-Enhancers
Check `baseline.py`, various kinds of embedding-visible LLMs (like LLaMA, SentenceBERT, or text-ada-embedding-002) can be used to generate embeddings as node features.
![LLMs-as-Enhancers](https://github.com/CurryTang/Graph-LLM/blob/master/imgs/llm_as_enhancer.png)

### (New project) LLMs-as-Annotators
Check out our new project here: [Label-free Node Classification on Graphs with Large Language Models (LLMS)
](https://github.com/CurryTang/LLMGNN) 


## Citation
```
@article{Chen2023ExploringTP,
  title={Exploring the Potential of Large Language Models (LLMs) in Learning on Graphs},
  author={Zhikai Chen and Haitao Mao and Hang Li and Wei Jin and Haifang Wen and Xiaochi Wei and Shuaiqiang Wang and Dawei Yin and Wenqi Fan and Hui Liu and Jiliang Tang},
  journal={ArXiv},
  year={2023},
  volume={abs/2307.03393}
}
```

## 0. Environment Setup

### Package Installation
Assume your cuda version is 11.8
```
conda create --name LLMGNN python=3.10
conda activate LLMGNN

conda install pytorch==2.0.0 cudatoolkit=11.8 -c pytorch
conda install -c pyg pytorch-sparse
conda install -c pyg pytorch-scatter
conda install -c pyg pytorch-cluster
conda install -c pyg pyg
pip install ogb
conda install -c dglteam/label/cu118 dgl
pip install transformers
pip install --upgrade accelerate
pip install openai
pip install langchain
pip install gensim
pip install google-generativeai
pip install -U sentence-transformers
pip install editdistance
pip install InstructorEmbedding
pip install optuna
pip install tiktoken
pip install pytorch_warmup
```

### Dataset 
We have provided the processed datasets via the following [google drive link](https://drive.google.com/drive/folders/1_laNA6eSQ6M5td2LvsEp3IL9qF6BC1KV?usp=sharing)

To unzip the files, you need to
1. unzip the `small_data.zip` into `preprocessed_data/new`
2. If you want to use ogbn-products, unzip `big_data.zip` info `preprocessed_data/new`
3. Download and move `*_explanation.pt` and `*_pl.pt` into `preprocessed_data/new`. These files are related to TAPE.
4. unzip the `ada.zip` into `./`
5. Move `*_entity.pt` into `./`
5. Put `ogb_arxiv.csv` into `./preprocessed_data`

### Get ft and no-ft LM embeddings

Refer to the following scripts
``` bash 
for setting in "random"
do 
    for data in "cora" "pubmed"
    do
        WANDB_DISABLED=True CUDA_VISIBLE_DEVICES=3 python3 lmfinetune.py --dataset $data --split $setting --batch_size=9 --label_smoothing 0.3 --seed_num 5 
        WANDB_DISABLED=True CUDA_VISIBLE_DEVICES=3 python3 lmfinetune.py --dataset $data --split $setting --batch_size=9 --label_smoothing 0.3 --seed_num 5 --use_explanation 1
    done
done
```

### Generate pt files for all data formats
Run 
``` python
python3 generate_pyg_data.py
```



## 1. Experiments for **LLM-as-Enhancers**

For feature-level, **LLM-as-Enhancers**, you may replicate the experiments using files **baseline.py** and **lmfinetune.py**

For example, you may run param sweep with the following script
``` bash
for model in "GCN" "GAT" "MLP"
do
    for data in "cora" "pubmed"
    do 
        for setting in "random"
        do 
        # Add more formats here
            for format in "ft"
            do 
                CUDA_VISIBLE_DEVICES=1 python3 baseline.py --model_name $model  --seed_num 5 --sweep_round 40  --mode sweep --dataset $data --split $setting --data_format $format
                echo "$model $data $setting $format done"
            done
        done
    done
done
```

Run with a specific group of hyperparameters
``` bash
python3 baseline.py --data_format sbert --split random --dataset pubmed --lr 0.01 --seed_num 5
```

Feature ensemble, separate each ensemble format with "\;"
``` bash
CUDA_VISIBLE_DEVICES=1 python3 baseline.py --model_name GCN --num_split 1 --seed_num 5 --sweep_split 1 --sweep_round 5 --mode sweep --dataset pubmed --split random --ensemble_string sbert\;know_sep_sb\;ft\;pl\;know_exp_ft
```

Batch version for ogbn-products
``` bash
CUDA_VISIBLE_DEVICES=7 python3 baseline.py --model_name SAGE --epochs 10 --num_split 1 --batchify 1  --dataset products --split fixed --data_format ft --normalize 1 --norm BatchNorm --mode main --lr 0.003 --dropout 0.5 --weight_decay 0 --hidden_dimension 256 --num_layers 3
``` 

To replicate the results for RevGAT (You need to first run once with the default features to generate the dgl data)
``` bash
python dgl_main.py --data_root_dir ./dgldata \
--pretrain_path  ./preprocessed_data/new/arxiv_fixed_sbert.pt \
--use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --save kd --backbone rev --group 2 --mode teacher


python dgl_main.py --data_root_dir ./dgldata \
--pretrain_path  ./preprocessed_data/new/arxiv_fixed_sbert.pt \
--use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 2 --dropout 0.75 --n-hidden 256 --save kd --backbone rev --group 2 --mode student --alpha 0.95 --temp 0.7
```

To replicate the results for [SAGN](https://github.com/THUDM/SCR/tree/main/ogbn-products) and [GLEM](https://github.com/AndyJZhao/GLEM), you may check their repositories and put the processed pt file into their pipelines.






## 2. Experiments for **LLM-as-Predictors**

Just run 
``` bash
python3 ego_graph.py
```



## 3. (UPDATE) Further Experiments on OOD & Prompts

In two recent studies titled [CAN LLMS EFFECTIVELY LEVERAGE GRAPH STRUCTURAL INFORMATION: WHEN AND WHY](https://arxiv.org/pdf/2309.16595.pdf) and [Explanations as Features: LLM-Based Features for Text-Attributed Graphs](https://arxiv.org/pdf/2305.19523), researchers probed a specific prompt tailored for the Arxiv dataset containing data from post-2023, data which ChatGPT's pre-training corpus doesn't cover. Notably, the results showed no decline in performance compared to the original dataset. This intriguing outcome prompts us to delve deeper into creating efficacious prompts across varied domains.

Out-of-distribution (OOD) generalization, commonly known as Graph OOD, is a fervent area of discussion. Recent benchmarks, such as [GOOD](https://github.com/divelab/GOOD/tree/GOODv1/GOOD), indicate that GNNs don't fare well during structural and feature shifts. We embarked on an experiment using the Arxiv dataset to assess the potential of LLMs-as-Predictors, leveraging a prompt from [Explanations as Features: LLM-Based Features for Text-Attributed Graphs](https://arxiv.org/pdf/2305.19523), which exhibited superior performance.

|                  	| All avg      	| Val   	| Test     	| Best baseline (test) 	|
|------------------	|--------------	|-------	|----------	|----------------------	|
| concept degree   	| 73.91 ± 0.63 	| 73.01 	| 72.79    	| 63.00                	|
| covariate degree 	| 75.75 ± 3.6  	| 70.23 	| 68.21    	| 59.08                	|
| concept time     	| 74.29 ± 0.96 	| 72.66 	| 71.98    	| 67.45                	|
| covariate time   	| 72.69 ± 1.53 	| 74.28 	| 74.37    	| 71.34                	|

* **Concept-shift**: Where P(Y|X) varies, yet its construct remains anchored to covariate-shift by adjusting the ratios in each domain.
* **Covariate-shift**: While P(X) shifts, P(Y|X) remains consistent.

For the covariate shift, there are configurations of 10/1/1 environments (train/val/test), and for the concept shift, it's 3/1/1 (train/val/test). The term `All avg` represents the mean performance across all environments.

One discernible merit of using LLMs-as-Predictors is their heightened resilience to OOD shifts.

