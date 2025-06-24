import os
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm
import numpy as np
from datasets import Dataset, DatasetDict
import logging
import sys
from datetime import datetime
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
      # Model parameters
    parser.add_argument("--model_name", type=str, default="codebert", help="Model name")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model")
    
    # Data parameters
    parser.add_argument("--train_file", type=str, default=None, help="Path to training data")
    parser.add_argument("--dataset_type", type=str, default="no_comments", choices=["comments", "no_comments"], help="Dataset type: 'comments' or 'no_comments'")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=8, help="Validation batch size")
    parser.add_argument("--accumulation_steps", type=int, default=32, help="Gradient accumulation steps")
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--log_file_name", type=str, default="log", help="Log file name")
    
    return parser.parse_args()


def data_preprocessing(row, tokenizer, max_length):
    return tokenizer(row['code'], truncation=True, max_length=max_length)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def main():
    args = parse_args()
      # Set file prefix based on dataset type
    if args.dataset_type == "comments":
        file_prefix = "fine-tune-comments"
    else:
        file_prefix = "fine-tune-NOcomments"

    now = datetime.now()
    date_string = now.strftime("%Y%m%d%H%M%S") 

    # Setup logging
    log_file_path = os.path.join(args.output_dir, args.model_name, f"{file_prefix}", f"{args.log_file_name}_{date_string}.txt")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logging.basicConfig(
        filename=log_file_path,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.info("Starting training...")
      # Print parameter information
    content_write = "=" * 50 + "\n"
    content_write += f"model_name: {args.model_name}\n"
    content_write += f"model_path: {args.model_path}\n"
    content_write += f"train_file: {args.train_file}\n"
    content_write += f"dataset_type: {args.dataset_type}\n"
    content_write += f"seed: {args.seed}\n"
    content_write += f"max_seq_length: {args.max_seq_length}\n"
    content_write += f"accumulation_steps: {args.accumulation_steps}\n"
    content_write += f"batch_size: {args.batch_size}\n"
    content_write += f"val_batch_size: {args.val_batch_size}\n"
    content_write += f"num_epochs: {args.num_epochs}\n"
    content_write += f"learning_rate {args.learning_rate:.0e}\n"
    content_write += f"weight_decay {args.weight_decay:.0e}\n"
    content_write += f"output_dir: {args.output_dir}\n"
    content_write += f"log_file_name: {args.log_file_name}\n"
    content_write += "=" * 50 + "\n"
    print(content_write)
    logging.info(content_write)

    # Load data
    data = pd.read_csv(args.train_file)
    data = data.dropna(subset=['func_before', 'func_after'])

    buggy_codes = data['func_before'].tolist()  # label 1
    clean_codes = data['func_after'].tolist()  # label 0

    df_list1 = pd.DataFrame({'code': buggy_codes, 'label': 1})
    df_list0 = pd.DataFrame({'code': clean_codes, 'label': 0})

    # Merge DataFrames
    data = pd.concat([df_list1, df_list0], ignore_index=True)

    # Shuffle data
    data = data.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Split data into train/val/test sets
    train_df = data[:int(len(data) * 0.8)]
    val_df = data[int(len(data) * 0.8):int(len(data) * 0.9)]
    test_df = data[int(len(data) * 0.9):]

    # Convert to HuggingFace dataset format
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    my_dataset = DatasetDict({
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    })

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=2
    )
    
    if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Data preprocessing
    def preprocess_function(examples):
        return data_preprocessing(examples, tokenizer, args.max_seq_length)

    tokenized_data = my_dataset.map(preprocess_function, batched=True, remove_columns=['code'])

    # Setup training parameters
    model_output_path = os.path.join(args.output_dir, args.model_name, f"{file_prefix}")

    training_args = TrainingArguments(
        output_dir=model_output_path,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=0.1,
        
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        
        num_train_epochs=args.num_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        
        save_total_limit=2,
        seed=args.seed,
        data_seed=args.seed,
        
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model()
    
    print("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_data["test"])
    print(f"Test results: {test_results}")
    logging.info(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
