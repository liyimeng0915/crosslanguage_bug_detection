import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, classification_report
import numpy as np
import os
import logging
from datetime import datetime
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="codebert", help="Model name")
    parser.add_argument("--model_path", type=str, default=None, help="Model path")
    parser.add_argument("--test_model", type=str, default="epoch_1", help="Test model checkpoint to use")

    parser.add_argument("--data_path", type=str, default=None, help="Data path")
    parser.add_argument("--comments", action="store_true", default=False, help="Whether to use comments in the data")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for testing")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--output_dir", type=str, default="output_codenet", help="Output directory")
    parser.add_argument("--should_log", action="store_true", default=True, help="Whether to enable logging")
    parser.add_argument("--log_file_name", type=str, default="log_test", help="Log file name")
    
    return parser.parse_args()


class CodeBugDataset(Dataset):
    def __init__(self, buggy_codes, clean_codes, tokenizer, max_length):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        for code in buggy_codes:
            self.data.append((code, 1))
        
        for code in clean_codes:
            self.data.append((code, 0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code, label = self.data[idx]
        
        inputs = self.tokenizer(
            code,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            probs = torch.softmax(logits, dim=-1)[:, 1] 
            all_probs.extend(probs.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    report = classification_report(all_labels, all_preds, target_names=['No Bug', 'Bug'])

    return accuracy, f1, precision, recall, auc, report


def main():
    args = parse_args()
    
    # Determine file prefix
    if args.comments:
        file_prefix = "test_comments"
    else:
        file_prefix = "test_NOcomments"
    
    # Set up logging
    now = datetime.now()
    date_string = now.strftime("%Y%m%d%H%M%S")
    
    logger = logging.getLogger(__name__)
    log_file_path = os.path.join(args.output_dir, args.model_name, file_prefix, f"{args.log_file_name}_{args.test_model}.txt")
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
    logger.setLevel(logging.INFO if args.should_log else logging.WARN)
    
    # Log parameters
    content_write = "=" * 50 + "\n"
    content_write += f"model_name: {args.model_name}\n"
    content_write += f"model_path: {args.model_path}\n"
    content_write += f"data_path: {args.data_path}\n"
    content_write += f"seed: {args.seed}\n"
    content_write += f"max_seq_length: {args.max_seq_length}\n"
    content_write += f"batch_size: {args.batch_size}\n"
    content_write += f"output_dir: {args.output_dir}\n"
    content_write += f"should_log: {args.should_log}\n"
    content_write += f"log_file_name: {args.log_file_name}\n"
    content_write += "=" * 50 + "\n"
    print(content_write)
    logger.info(content_write)
    
    # Set random seed
    set_seed(args.seed)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=2)
    
    # Load data
    data = pd.read_csv(args.data_path)
    data = data.dropna(subset=['func_before', 'func_after'])

    buggy_codes = data['func_before'].tolist()
    clean_codes = data['func_after'].tolist()
    
    print(f"Total buggy codes: {len(buggy_codes)}")
    print(f"Total clean codes: {len(clean_codes)}")
    
    # Split dataset
    train_buggy_codes, temp_buggy_codes, train_clean_codes, temp_clean_codes = train_test_split(
        buggy_codes, clean_codes, test_size=0.2, random_state=args.seed
    )

    val_buggy_codes, test_buggy_codes, val_clean_codes, test_clean_codes = train_test_split(
        temp_buggy_codes, temp_clean_codes, test_size=0.5, random_state=args.seed
    )
    
    print(f"Train set: {len(train_buggy_codes)} buggy, {len(train_clean_codes)} clean")
    print(f"Val set: {len(val_buggy_codes)} buggy, {len(val_clean_codes)} clean")
    print(f"Test set: {len(test_buggy_codes)} buggy, {len(test_clean_codes)} clean")
    
    # Create test dataset and dataloader
    test_dataset = CodeBugDataset(test_buggy_codes, test_clean_codes, tokenizer, max_length=args.max_seq_length)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    val_accuracy, val_f1, val_precision, val_recall, val_auc, val_report = evaluate_model(model, test_dataloader, device)

    val_res = "=" * 20 + "Test_res" + "=" * 20 + "\n"
    val_res += f"Accuracy:  {val_accuracy:.4f}\n"
    val_res += f"Precision: {val_precision:.4f}\n"
    val_res += f"Recall:    {val_recall:.4f}\n"
    val_res += f"F1 Score:  {val_f1:.4f}\n"
    val_res += f"AUC:       {val_auc:.4f}\n"
    val_res += f"Classification Report:\n{val_report}\n"
    val_res += "=" * 50 + "\n"
    print(val_res)
    logger.info(val_res)


if __name__ == "__main__":
    main()
