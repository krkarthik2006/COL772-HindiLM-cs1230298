import os
import math 
import re
from time import time

import torch
from torch.optim.lr_scheduler import LambdaLR 

from partb.bpe_tokenizer import BPETokenizer
from parta.model import LanguageModel, collate_fn
from torch.utils.data import Dataset, DataLoader
from .utils import calculate_bpc


def collate_batch_for_lm(batch):
    batch_dict = {
        "input_ids": [item["input_ids"] for item in batch],
        "attention_mask": [item["attention_mask"] for item in batch],
    }
    return collate_fn(batch_dict)


def build_scheduler(optimizer, num_warmup_steps, num_training_steps):
    """Linear warmup followed by cosine decay."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def find_latest_checkpoint(checkpoint_dir):
    """Return latest epoch checkpoint path in checkpoint_dir, or None."""
    if not os.path.isdir(checkpoint_dir):
        return None

    pattern = re.compile(r"^checkpoint_epoch_(\d+)\.pt$")
    latest_epoch = -1
    latest_path = None

    for filename in os.listdir(checkpoint_dir):
        match = pattern.match(filename)
        if match is None:
            continue
        epoch_num = int(match.group(1))
        if epoch_num > latest_epoch:
            latest_epoch = epoch_num
            latest_path = os.path.join(checkpoint_dir, filename)

    return latest_path

class dataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        token_ids = self.tokenizer.encode(text)
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(token_ids), dtype=torch.long)
        }

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    # if device.type == 'cuda' and torch.cuda.is_bf16_supported():
    #     ptdtype = torch.bfloat16
    #     print("AMP: Using native Bfloat16 (CUDA)")
    # elif device.type == 'cpu':
    #     ptdtype = torch.bfloat16 
    #     print("AMP: Using simulated Bfloat16 (CPU)")
    # else:
    #     ptdtype = torch.float16 
    #     print("AMP: Using standard Float16 (MPS/Fallback)")
        
    # use_scaler = (ptdtype == torch.float16)
    # scaler = torch.amp.GradScaler(enabled=use_scaler)
    # load tokenizer
    tokenizer = BPETokenizer()
    tokenizer_path = args.tokenizer_path
    tokenizer.load(tokenizer_path)
    
    # load datasets
    train_data = []
    with open(args.train_path, 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(line.strip())
    
    valid_data = []
    with open(args.valid_path, 'r', encoding='utf-8') as f:
        for line in f:
            valid_data.append(line.strip())
            
    train_dataset = dataset(train_data, tokenizer, max_length=args.max_seq_len)
    valid_dataset = dataset(valid_data, tokenizer, max_length=args.max_seq_len)
    print("Calculating validation character-to-token ratio for BPC...")
    total_valid_chars = sum(len(text) for text in valid_data)
    # Count tokens, capping at max_seq_len just like the dataset class does
    total_valid_tokens = sum(min(len(tokenizer.encode(text)), args.max_seq_len) for text in valid_data)
    print(f"Valid set has {total_valid_chars} chars and {total_valid_tokens} tokens.")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch_for_lm)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch_for_lm)
    
    config = {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "vocab_size": tokenizer.get_vocab_size(),
        "dropout": args.dropout,
        "attention_dropout": args.attention_dropout,
        "mode": "standard"
    }
    
    model = LanguageModel(config).to(device)
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    accumulation_steps = 4
    total_steps = (len(train_dataloader) // accumulation_steps) * args.epochs
    scheduler = build_scheduler(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    
    best_valid_loss = float('inf')
    start_epoch = 0
    os.makedirs(args.output_model_path, exist_ok=True)

    resume_path = None
    if args.resume_from is not None:
        resume_path = args.resume_from
    elif args.auto_resume:
        resume_path = find_latest_checkpoint(args.output_model_path)

    if resume_path is not None:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_valid_loss = checkpoint.get('best_valid_loss', checkpoint.get('loss', float('inf')))
        start_epoch = checkpoint.get('epoch', -1) + 1
        print(f"Resumed training from {resume_path} (starting at epoch {start_epoch + 1})")

    if start_epoch >= args.epochs:
        print(f"Checkpoint is already at/after target epochs ({args.epochs}). Nothing to train.")
        return
    
    print("Starting training...")
    accumulated_steps = 4
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for (step,batch) in enumerate(train_dataloader):
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # with torch.autocast(device_type=device.type, dtype=ptdtype):
            logits = model(input_ids, attention_mask)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            loss = criterion(shift_logits.view(-1, config['vocab_size']), shift_labels.view(-1))
            loss = loss / accumulated_steps
            loss.backward()
            if step % 10 == 0:
                print(f"Step {step} | Loss: {loss.item() * accumulation_steps:.4f}")
            if (step + 1) % accumulated_steps == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)#to 0.5?
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad() 
                
            
            total_loss += (loss.item() * accumulated_steps)
            
        avg_train_loss = total_loss / len(train_dataloader)
        
        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for batch in valid_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                logits = model(input_ids, attention_mask)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = criterion(shift_logits.view(-1, config['vocab_size']), shift_labels.view(-1))
                total_valid_loss += loss.item()
                
        avg_valid_loss = total_valid_loss / len(valid_dataloader)
        
        
        valid_bpc = calculate_bpc(avg_valid_loss, total_valid_tokens, total_valid_chars)
        print(
            f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f} "
            f"- Valid Loss: {avg_valid_loss:.4f}"
            f" - Valid BPC: {valid_bpc:.4f}"
        )
        
        if args.save_intermediate:
            checkpoint_path = os.path.join(args.output_model_path, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_valid_loss': best_valid_loss,
                'loss': avg_valid_loss,
            }, checkpoint_path)
            print(f"Saved intermediate checkpoint to {checkpoint_path}")

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_model_path = os.path.join(args.output_model_path, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Saved new best model to {best_model_path}")

if __name__ == '__main__':
    import argparse
    start_time = time()

    parser = argparse.ArgumentParser(description='Train a model on the given dataset.')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the train dataset')
    parser.add_argument('--valid_path', type=str, required=True, help='Path to the valid dataset')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer')
    parser.add_argument('--output_model_path', type=str, default='checkpoints', help='Directory to save checkpoints')
    
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Steps for learning rate warmup')
    
    parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum context window')
    parser.add_argument('--d_model', type=int, default=768, help='Dimension of the model')
    parser.add_argument('--n_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=12, help='Number of transformer layers')
    
    parser.add_argument('--dropout', type=float, default=0.15, help='General Dropout probability')
    parser.add_argument('--attention_dropout', type=float, default=0.15, help='Attention Dropout probability')

    parser.add_argument('--save_intermediate', action='store_true', help='Flag to save intermediate optimizer/model states')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to a checkpoint (.pt) to resume training from')
    parser.add_argument('--auto_resume', action='store_true', help='If set, resume from latest epoch checkpoint in output_model_path')
    
    args = parser.parse_args()
    main(args)
    
    end_time = time()
    elapsed = int(end_time - start_time)
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"Total training time: {hrs}h {mins}m {secs}s")
