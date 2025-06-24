# Cross-Language Bug Detection
This repository contains the code and datasets used in the experiments of our research paper—`Fine-Tuning Code Language Models to Detect Cross-Language Bugs`. It serves as a replication package to facilitate the reproduction of the results presented in the paper.

## Structure of the Repository

### `datasets` folder
Contains all experimental datasets for the four research questions (RQs):
- `RQ1`: Cross-language datasets (without code comments version)
- `RQ2`: CodeNet and CVEFixes datasets
- `RQ3`: Does not contain separate datasets. Instead, it reuses the cross-language datasets from RQ1 to analyze the impact of dataset size and token sequence
- `RQ4`: Cross-language datasets (with code comments version)

### `experiments` folder
This folder contains the scripts used in our experiments. Below are details of the subfolders:

- `RQ1`: Contains fine-tuning and testing scripts for 13 Code Language Models (CodeLMs).
- `RQ2`: Contains two subfolders (`codenet` and `cvefixes`) for fine-tuning and testing scripts related to CodeNet and CVEFixes datasets.
- `RQ3`: Contains two subfolders (`datasetSize` and `tokenLength`) for analyzing the impact of dataset size and token sequence length on CodeLMs. Each subfolder includes fine-tuning and testing scripts.
- `RQ4`: It uses the same scripts as the RQ1 folder. You can modify the shell commands to run experiments for RQ4.

### requirements files
- `requirements_4090D.txt`: Python packages for RTX 4090D GPU environment
- `requirements_L20.txt`: Python packages for L20 GPU environment

## Setting Up the Experiment Environment
1. Clone the repository:
```bash
git clone git@github.com:liyimeng0915/crosslanguage_bug_detection.git
```
2. Create a Python environment:
```bash
conda create -n crosslang_env python=3.8
conda activate crosslang_env
```
3. Install dependencies:
```bash
pip install -r requirements_4090D.txt  # or requirements_L20.txt based on your GPU
```

## Usage Examples

### Fine-tuning a CodeLM
Below is an example of fine-tuning the CodeBERT model for RQ1:

```bash
python codebert.py \
    --model_name "codebert" \
    --model_path "../models/codebert-base" \
    --train_file "../datasets/cross-language_datasets_without_comments.csv" \
    --dataset_type "no_comments" \
    --max_seq_length 512 \
    --batch_size 16 \
    --accumulation_steps 8 \
    --num_epochs 10 \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --seed 42 \
    --output_dir "output" \
    --log_file_name "log"
```

### Evaluating a Fine-tuned CodeLM
Below is an example of testing the fine-tuned CodeBERT model:

```bash
python test_codebert.py \
    --model_name "codebert" \
    --model_path "output/" \
    --test_model "epoch_10" \
    --data_path "../datasets/cross-language_datasets_without_comments.csv" \
    --dataset_type "no_comments" \
    --max_seq_length 512 \
    --batch_size 8 \
    --seed 42 \
    --output_dir "test_results" \
    --log_file_name "log_test"
```

**Note**: The above examples demonstrate the usage for the CodeBERT model in RQ1. All other models in different RQs have similar training and testing scripts. Please adjust the model names, dataset paths, and parameters according to your specific requirements. Each model may have different parameter configurations—refer to the individual script files in each folder for details.

