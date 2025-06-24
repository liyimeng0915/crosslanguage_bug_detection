import os
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, Trainer, TrainingArguments, set_seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import numpy as np
from datasets import Dataset, DatasetDict
import argparse
import logging
import sys
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
      # Model parameters
    parser.add_argument("--model_name", type=str, default="codebert", help="Model name")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint")
    
    # Data parameters
    parser.add_argument("--train_file", type=str, default=None, help="Path to training data")
    parser.add_argument("--dataset_type", type=str, default="no_comments", choices=["comments", "no_comments"], help="Dataset type: 'comments' or 'no_comments'")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="Gradient accumulation steps")
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use fp16")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_total_limit", type=int, default=10, help="Save total limit")
    
    args = parser.parse_args()
    
    return args


def data_preprocessing(row, tokenizer, max_length):
    return tokenizer(row['code'], truncation=True, max_length=max_length)


def compute_metrics(eval_pred):
    # Unpack predictions and labels
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Handle multi-dimensional predictions
    if predictions.ndim > 1:
        preds = predictions.argmax(axis=-1)
        auc = roc_auc_score(labels, predictions[:, 1])
    else:
        preds = (predictions > 0.5).astype(int)
        auc = roc_auc_score(labels, predictions)

    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    f1 = f1_score(labels, preds, average='binary')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
    }


def main():
    args = parse_args()
    
    # 根据数据集类型设置输出目录
    if args.dataset_type == "comments":
        output_dir = f"{args.output_dir}/{args.model_name}/fine-tune-comments"
        logging_dir = f"{args.model_name}_comments"
    else:
        output_dir = f"{args.output_dir}/{args.model_name}/fine-tune-NOcomments"
        logging_dir = f"{args.model_name}_no_comments"

    # 设置随机种子
    set_seed(args.seed)
    data = pd.read_csv(args.train_file)  

    # Remove null values
    data = data.dropna(subset=['func_before', 'func_after'])

    buggy_codes = data['func_before'].tolist()  # label 1
    clean_codes = data['func_after'].tolist()   # label 0

    df_list1 = pd.DataFrame({'code': buggy_codes, 'label': 1})
    df_list0 = pd.DataFrame({'code': clean_codes, 'label': 0})

    # Merge DataFrames
    data = pd.concat([df_list1, df_list0], ignore_index=True)

    # Shuffle data
    data = data.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Split data into train/val/test sets
    train_df, val_df, test_df = data[:int(len(data) * 0.7)], data[int(len(data) * 0.7):int(len(data) * 0.85)], data[int(len(data) * 0.85):]

    # Convert to HuggingFace dataset format
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    my_dataset = DatasetDict({
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    })

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=2)
    model.config.pad_token_id = 0
    model.config.pad_token_id = model.config.eos_token_id

    # Setup tokenizer
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = '0'
    tokenizer.padding_side = "right"

    # Data preprocessing
    def preprocess_fn(row):
        return data_preprocessing(row, tokenizer, args.max_seq_length)

    tokenized_data = my_dataset.map(preprocess_fn, batched=True, remove_columns=['code'])
    tokenized_data.set_format("torch")

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,

        do_train=True,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        logging_dir=logging_dir,
        
        fp16=args.fp16,
        
        evaluation_strategy='epoch',
        save_strategy='epoch',
        
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        group_by_length=True, 
        report_to="none"
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['val'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Start training
    try:
        if args.resume_from_checkpoint:
            print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            trainer.train()
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
