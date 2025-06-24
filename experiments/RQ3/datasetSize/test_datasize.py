import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm
import numpy as np
import os
import logging
from datetime import datetime
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="codebert", help="Model name for output directory")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the fine-tuned model")
    parser.add_argument("--test_model", type=str, default="epoch_1", help="Specific model checkpoint to test")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default="../datasets/merged_all_2_3_dedupliacted2_dropna3_dropeq.csv", help="Path to the dataset CSV file")
    parser.add_argument("--comments", type=bool, default=True, help="Whether to use dataset with comments")
    parser.add_argument("--data_ratio", type=float, default=0.2, help="Ratio of data to use from the dataset")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Testing batch size")
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="", help="Output directory")
    parser.add_argument("--should_log", type=bool, default=True, help="Whether to enable logging")
    parser.add_argument("--log_file_name", type=str, default="", help="Log file name")
    
    args = parser.parse_args()
    
    # Auto-generate output_dir and log_file_name if not provided
    if not args.output_dir:
        args.output_dir = f"output_datasize_{args.data_ratio}"
    if not args.log_file_name:
        args.log_file_name = f"log_test_datasize_{args.data_ratio}"
    
    return args


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
        
        # Tokenize code
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

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            probs = torch.softmax(logits, dim=-1)[:, 1] 
            all_probs.extend(probs.cpu().numpy())

    # Calculate AUC score
    auc = roc_auc_score(all_labels, all_probs)

    # Calculate other evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=['No Bug', 'Bug'])

    return accuracy, f1, precision, recall, auc, report


def main():
    args = parse_args()
    
    # Set file prefix based on comments setting
    file_prefix = "test_comments" if args.comments else "test_NOcomments"
    
    # Generate timestamp
    now = datetime.now()
    date_string = now.strftime("%Y%m%d%H%M%S")
    
    # Setup logging
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
    
    # Log configuration
    content_write = "=" * 50 + "\n"
    content_write += "说明：测试\n"
    content_write += f"model_name: {args.model_name}\n"
    content_write += f"model_path: {args.model_path}\n"
    content_write += f"data_path: {args.data_path}\n"
    content_write += f"seed: {args.seed}\n"
    content_write += f"max_seq_length: {args.max_seq_length}\n"
    content_write += f"batch_size: {args.batch_size}\n"
    content_write += f"output_dir: {args.output_dir}\n"
    content_write += f"should_log: {args.should_log}\n"
    content_write += f"log_file_name: {args.log_file_name}\n"
    content_write += f"data_ratio: {args.data_ratio}\n"
    content_write += "=" * 50 + "\n"
    print(content_write)
    logger.info(content_write)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=2)
    
    # Set seeds
    np.random.seed(args.seed)
    
    # Load and prepare data
    data = pd.read_csv(args.data_path)
    data = data.dropna(subset=['func_before', 'func_after'])
    data = data.sample(frac=args.data_ratio, random_state=args.seed)

    buggy_codes = data['func_before'].tolist()
    clean_codes = data['func_after'].tolist()

    print(f"Buggy codes: {len(buggy_codes)}")
    print(f"Clean codes: {len(clean_codes)}")

    # Split data (same as training split)
    train_buggy_codes, temp_buggy_codes, train_clean_codes, temp_clean_codes = train_test_split(
        buggy_codes, clean_codes, test_size=0.2, random_state=args.seed
    )

    val_buggy_codes, test_buggy_codes, val_clean_codes, test_clean_codes = train_test_split(
        temp_buggy_codes, temp_clean_codes, test_size=0.5, random_state=args.seed
    )

    print(f"Train - Buggy: {len(train_buggy_codes)}, Clean: {len(train_clean_codes)}")
    print(f"Val - Buggy: {len(val_buggy_codes)}, Clean: {len(val_clean_codes)}")
    print(f"Test - Buggy: {len(test_buggy_codes)}, Clean: {len(test_clean_codes)}")

    # Create test dataset and dataloader
    test_dataset = CodeBugDataset(test_buggy_codes, test_clean_codes, tokenizer, max_length=args.max_seq_length)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluate model
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
