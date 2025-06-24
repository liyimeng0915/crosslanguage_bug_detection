import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, set_seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm
import numpy as np
import os
import time 
import logging
from datetime import datetime
import sys
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="codebert", help="Model name")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Path to pretrained model")

    parser.add_argument("--train_java_true_file", type=str, default="../datasets/cvefixes_java_true_drop_comments.csv", help="Java true training data file")
    parser.add_argument("--train_java_false_file", type=str, default="../datasets/cvefixes_java_false_drop_comments.csv", help="Java false training data file")
    parser.add_argument("--train_python_true_file", type=str, default="../datasets/cvefixes_python_true_drop_comments.csv", help="Python true training data file")
    parser.add_argument("--train_python_false_file", type=str, default="../datasets/cvefixes_python_false_drop_comments.csv", help="Python false training data file")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--accumulation_steps", type=int, default=32, help="Gradient accumulation steps")
    parser.add_argument("--max_java_false_samples", type=int, default=3102, help="Maximum number of Java false samples")
    parser.add_argument("--max_python_false_samples", type=int, default=2775, help="Maximum number of Python false samples")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")

    parser.add_argument("--output_dir", type=str, default="output_cvefixes", help="Output directory")
    parser.add_argument("--file_prefix", type=str, default="fine-tune_cvefixes", help="File prefix")
    parser.add_argument("--should_log", type=bool, default=True, help="Whether to log")
    parser.add_argument("--log_file_name", type=str, default="log", help="Log file name")
    
    return parser.parse_args()


class CodeBugDataset(Dataset):
    def __init__(self, buggy_codes, clean_codes, tokenizer, max_length=512):
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


def setup_logging(args, date_string):

    logger = logging.getLogger(__name__)

    log_file_path = os.path.join(args.output_dir, args.model_name, args.file_prefix, f"{args.log_file_name}_{date_string}.txt")
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
    
    return logger


def log_parameters(args, logger):
    content_write = "=" * 50 + "\n"
    content_write += f"model_name: {args.model_name}\n"
    content_write += f"model_name_or_path: {args.model_name_or_path}\n"
    content_write += f"seed: {args.seed}\n"
    content_write += f"max_seq_length: {args.max_seq_length}\n"
    content_write += f"accumulation_steps: {args.accumulation_steps}\n"
    content_write += f"batch_size: {args.batch_size}\n"
    content_write += f"num_epochs: {args.num_epochs}\n"
    content_write += f"learning_rate {args.learning_rate:.0e}\n"
    content_write += f"weight_decay {args.weight_decay:.0e}\n"
    content_write += f"output_dir: {args.output_dir}\n"
    content_write += f"should_log: {args.should_log}\n"
    content_write += f"log_file_name: {args.log_file_name}\n"
    content_write += "=" * 50 + "\n"
    print(content_write)
    logger.info(content_write)


def load_data(args):

    df_java_true = pd.read_csv(args.train_java_true_file)
    df_java_false = pd.read_csv(args.train_java_false_file)
    df_python_true = pd.read_csv(args.train_python_true_file)
    df_python_false = pd.read_csv(args.train_python_false_file)

    buggy_codes_java = df_java_true.iloc[:, 0].tolist()
    buggy_codes_python = df_python_true.iloc[:, 0].tolist()
    buggy_codes = buggy_codes_java + buggy_codes_python

    clean_codes_java = df_java_false.iloc[:, 0].tolist()
    clean_codes_python = df_python_false.iloc[:, 0].tolist()

    random.seed(args.seed)
    random.shuffle(clean_codes_java)
    random.shuffle(clean_codes_python)

    # 限制数据量
    clean_codes_java = clean_codes_java[:args.max_java_false_samples]
    clean_codes_python = clean_codes_python[:args.max_python_false_samples]
    clean_codes = clean_codes_java + clean_codes_python
    
    print(f"Loaded {len(buggy_codes)} buggy codes and {len(clean_codes)} clean codes")
    
    return buggy_codes, clean_codes


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = [] 
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = outputs.loss

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    auc = roc_auc_score(all_labels, all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    report = classification_report(all_labels, all_preds, target_names=['No Bug', 'Bug'])

    return avg_loss, accuracy, f1, precision, recall, auc, report


def main():

    args = parse_args()

    now = datetime.now()
    date_string = now.strftime("%Y%m%d%H%M%S")

    logger = setup_logging(args, date_string)

    log_parameters(args, logger)

    set_seed(args.seed)

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=2)
    
    buggy_codes, clean_codes = load_data(args)

    train_buggy_codes, temp_buggy_codes, train_clean_codes, temp_clean_codes = train_test_split(
        buggy_codes, clean_codes, test_size=0.2, random_state=args.seed
    )

    val_buggy_codes, test_buggy_codes, val_clean_codes, test_clean_codes = train_test_split(
        temp_buggy_codes, temp_clean_codes, test_size=0.5, random_state=args.seed
    )
    
    print(f"Train: {len(train_buggy_codes)} buggy, {len(train_clean_codes)} clean")
    print(f"Val: {len(val_buggy_codes)} buggy, {len(val_clean_codes)} clean")
    print(f"Test: {len(test_buggy_codes)} buggy, {len(test_clean_codes)} clean")
    
    val_dataset = CodeBugDataset(val_buggy_codes, val_clean_codes, tokenizer, max_length=args.max_seq_length)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    train_dataset = CodeBugDataset(train_buggy_codes, train_clean_codes, tokenizer, max_length=args.max_seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    

    print("Evaluating pre-trained model...")
    val_loss, val_accuracy, val_f1, val_precision, val_recall, val_auc, val_report = evaluate_model(model, val_dataloader, device)
    
    val_res = "=" * 20 + "Val_res (Pre-trained)" + "=" * 20 + "\n"
    val_res += f"Val_Loss:  {val_loss:.4f}\n"
    val_res += f"Accuracy:  {val_accuracy:.4f}\n"
    val_res += f"Precision: {val_precision:.4f}\n"
    val_res += f"Recall:    {val_recall:.4f}\n"
    val_res += f"F1 Score:  {val_f1:.4f}\n"
    val_res += f"AUC:       {val_auc:.4f}\n"
    val_res += f"Classification Report:\n{val_report}\n"
    val_res += "=" * 50 + "\n"
    print(val_res)
    logger.info(val_res)
    

    total_steps = len(train_dataloader) * args.num_epochs
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    
    best_validation_loss = float("inf")
    

    print("Starting fine-tuning...")
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss = loss / args.accumulation_steps
            loss.backward()
            
            if (step + 1) % args.accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_dataloader)
        train_loss_output = f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {avg_loss:.4f}"
        print(train_loss_output)
        logger.info(train_loss_output)
        
        val_loss, val_accuracy, val_f1, val_precision, val_recall, val_auc, val_report = evaluate_model(model, val_dataloader, device)

        val_res = "=" * 20 + f"Val_res_Epoch {epoch + 1}" + "=" * 20 + "\n"
        val_res += f"Epoch {epoch + 1} - Val_Loss:  {val_loss:.4f}\n"
        val_res += f"Epoch {epoch + 1} - Accuracy:  {val_accuracy:.4f}\n"
        val_res += f"Epoch {epoch + 1} - Precision: {val_precision:.4f}\n"
        val_res += f"Epoch {epoch + 1} - Recall:    {val_recall:.4f}\n"
        val_res += f"Epoch {epoch + 1} - F1 Score:  {val_f1:.4f}\n"
        val_res += f"Epoch {epoch + 1} - AUC:       {val_auc:.4f}\n"
        val_res += f"Epoch {epoch + 1} - Classification Report:\n{val_report}\n"
        val_res += "=" * 50 + "\n"
        print(val_res)
        logger.info(val_res)

        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_model_path = os.path.join(args.output_dir, args.model_name, args.file_prefix, f"best_model_{date_string}")
            os.makedirs(best_model_path, exist_ok=True)
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"Best model saved to {best_model_path}")

        # 保存每个epoch的模型
        save_path = os.path.join(args.output_dir, args.model_name, args.file_prefix, f"epoch_{epoch + 1}")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
    
    print("Training completed!")


if __name__ == "__main__":
    main()
