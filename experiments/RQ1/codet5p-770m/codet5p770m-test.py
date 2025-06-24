import os
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import numpy as np
from datasets import Dataset, DatasetDict
import argparse
import logging
import sys
from datetime import datetime
import glob


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default="codebert", help="Model name")
    parser.add_argument("--model_path", type=str, default=None, help="Path to base model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint to test")

    parser.add_argument("--data_path", type=str, default=None, help="Path to test data")
    parser.add_argument("--dataset_type", type=str, default="no_comments", choices=["comments", "no_comments"], help="Dataset type: 'comments' or 'no_comments'")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Evaluation batch size")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--log_file_name", type=str, default="test_res", help="Log file name prefix")
    
    args = parser.parse_args()
    
    if args.dataset_type == "comments":
        args.file_prefix = "fine-tune-comments"
        args.log_file_path = f"{args.log_file_name}_comments.txt"
    else:
        args.file_prefix = "fine-tune-NOcomments"
        args.log_file_path = f"{args.log_file_name}_NOcomments.txt"
    
    return args


def data_preprocessing(row, tokenizer, max_length):
    return tokenizer(row['code'], truncation=True, max_length=max_length)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    if predictions.ndim > 1:
        preds = predictions.argmax(axis=-1)
        auc = roc_auc_score(labels, predictions[:, 1])
    else:
        # Use threshold 0.5 for single dimension probability
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
    logging.basicConfig(
        filename=args.log_file_path,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.info("Starting testing...")

    set_seed(args.seed)

    # Load data
    data = pd.read_csv(args.data_path)  
    logging.info(f"args_data_path:{args.data_path}")
    logging.info(f"args_model_path:{args.model_path}")
    logging.info(f"args_max_seq_length:{args.max_seq_length}")

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

    print(f"Test dataset size: {len(my_dataset['test'])}")

    if args.checkpoint:
        checkpoint_path = f"{args.output_dir}/{args.model_name}/{args.file_prefix}/{args.checkpoint}"
        print(f"Evaluating specific checkpoint: {checkpoint_path}")
        logging.info(f"Evaluating specific checkpoint: {checkpoint_path}")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=2)
        model.config.pad_token_id = 0
        model.config.pad_token_id = model.config.eos_token_id

        # Setup tokenizer
        if '[PAD]' not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = '0'
        tokenizer.padding_side = "right"

        def preprocess_fn(row):
            return data_preprocessing(row, tokenizer, args.max_seq_length)

        tokenized_data = my_dataset.map(preprocess_fn, batched=True, remove_columns=['code'])
        tokenized_data.set_format("torch")
        
        evaluate_model(model, tokenized_data['test'], tokenizer)
    else:
        # Get all checkpoint paths and test them
        checkpoints = sorted(glob.glob(f"{args.output_dir}/{args.model_name}/{args.file_prefix}/checkpoint-*"))
        print(f"Found checkpoints: {checkpoints}")
        
        for checkpoint in checkpoints:
            print(f"\nEvaluating checkpoint: {checkpoint}")
            logging.info(f"\nEvaluating checkpoint: {checkpoint}")
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
            model.config.pad_token_id = 0
            model.config.pad_token_id = model.config.eos_token_id

            # Setup tokenizer
            if '[PAD]' not in tokenizer.get_vocab():
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.pad_token = '0'
            tokenizer.padding_side = "right"

            # Data preprocessing
            def preprocess_fn(row):
                return data_preprocessing(row, tokenizer, args.max_seq_length)

            tokenized_data = my_dataset.map(preprocess_fn, batched=True, remove_columns=['code'])
            tokenized_data.set_format("torch")

            evaluate_model(model, tokenized_data['test'], tokenizer)


if __name__ == "__main__":
    main()
