import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, Trainer, TrainingArguments, set_seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import numpy as np
from datasets import Dataset, DatasetDict
import csv
import logging
import sys
from datetime import datetime
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    TaskType,
    PeftModel
)
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="codebert", help="Model name")
    parser.add_argument("--model_path", type=str, default=None, help="Path to base model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    
    parser.add_argument("--data_path", type=str, default=None, help="Path to test data")
    parser.add_argument("--dataset_type", type=str, default="no_comments", choices=["comments", "no_comments"], help="Dataset type: 'comments' or 'no_comments'")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--val_batch_size", type=int, default=8, help="Validation batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA r")
    
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--log_file_name", type=str, default="log", help="Log file name")
    
    args = parser.parse_args()
    
    if args.dataset_type == "comments":
        args.file_prefix = "test-comments"
    else:
        args.file_prefix = "test-NOcomments"
    
    args.accumulation_steps = int(64 / args.batch_size)
    
    return args


def data_preprocessing(row, tokenizer, max_seq_length):
    return tokenizer(row['code'], truncation=True, max_length=max_seq_length)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1]  # 选择类别 1 的概率
    auc = roc_auc_score(labels, probabilities.numpy())

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def evaluate_model(model, test_dataset, tokenizer):
    model.eval()
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    results = trainer.evaluate(eval_dataset=test_dataset)
    print("Evaluation Results:", results)
    logging.info(f"Evaluation Results:{results}")
    logging.info("="*100)
    return results


def main():
    args = parse_args()
    
    now = datetime.now()
    date_string = now.strftime("%Y%m%d%H%M%S") 

    logger = logging.getLogger(__name__)
    log_file_path = os.path.join(args.output_dir, args.model_name, f"{args.file_prefix}", f"{args.log_file_name}_{date_string}.txt")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file_path)
        ]
    )

    logging.getLogger().handlers[0].setLevel(logging.WARNING)
    logger.setLevel(logging.INFO)
    
    content_write = "=" * 50 + "\n"
    content_write += "说明：测试\n"
    content_write += f"model_name: {args.model_name}\n"
    content_write += f"model_path: {args.model_path}\n"
    content_write += f"checkpoint: {args.checkpoint}\n"
    content_write += f"data_path: {args.data_path}\n"
    content_write += f"dataset_type: {args.dataset_type}\n"
    content_write += f"seed: {args.seed}\n"
    content_write += f"max_seq_length: {args.max_seq_length}\n"
    content_write += f"batch_size: {args.batch_size}\n"
    content_write += f"val_batch_size: {args.val_batch_size}\n"
    content_write += f"lora_alpha: {args.lora_alpha}\n"
    content_write += f"lora_dropout: {args.lora_dropout}\n"
    content_write += f"lora_r: {args.lora_r}\n"
    content_write += "=" * 50 + "\n"
    print(content_write)
    logger.info(content_write)

    set_seed(args.seed)

    data = pd.read_csv(args.data_path)  
    data = data.dropna(subset=['func_before', 'func_after'])

    buggy_codes = data['func_before'].tolist()
    clean_codes = data['func_after'].tolist() 

    df_list1 = pd.DataFrame({'code': buggy_codes, 'label': 1})
    df_list0 = pd.DataFrame({'code': clean_codes, 'label': 0})

    data = pd.concat([df_list1, df_list0], ignore_index=True)

    data = data.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    train_df, val_df, test_df = data[:int(len(data) * 0.8)], data[int(len(data) * 0.8):int(len(data) * 0.9)], data[int(len(data) * 0.9):]

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    my_dataset = DatasetDict({
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    })

    print(f"Test dataset size: {len(my_dataset['test'])}")

    if os.path.exists(args.checkpoint):
        print(f"Loading trained model from {args.checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, num_labels=2)
    else:
        print(f"Loading base model from {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            num_labels=2,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        model = get_peft_model(model, lora_config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def preprocess_fn(row):
        return data_preprocessing(row, tokenizer, args.max_seq_length)

    tokenized_data = my_dataset.map(preprocess_fn, batched=True, remove_columns=['code'])
    tokenized_data.set_format("torch")

    evaluate_model(model, tokenized_data['test'], tokenizer)


if __name__ == "__main__":
    main()
