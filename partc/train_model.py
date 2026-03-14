# YOUR TOKENIZER AND MODEL from PART A AND PART B RESPECTIVELY
# If you wish to change their code, please do so in their respective files under parta/ and partb/ directories.
import os
from time import time

import torch

from partb.bpe_tokenizer import BPETokenizer
from parta.model import LanguageModel, collate_fn
from torch.utils.data import Dataset, DataLoader
from .utils import calculate_perplexity


def collate_batch_for_lm(batch):
    batch_dict = {
        "input_ids": [item["input_ids"] for item in batch],
        "attention_mask": [item["attention_mask"] for item in batch],
    }
    return collate_fn(batch_dict)


class dataset(Dataset):
    def __init__(self, data,tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        token_ids = self.tokenizer.encode(text)
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(token_ids), dtype=torch.long)
        }

def main(args):
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    #load tokenizer
    tokenizer = BPETokenizer()
    tokenizer_path = args.tokenizer_path
    if os.path.isdir(tokenizer_path):
        tokenizer_path = os.path.join(tokenizer_path, "my_bpe.json")
    tokenizer.load(tokenizer_path)
    #load datasets
    train_data = []
    with open(args.train_path, 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(line.strip())
    
    valid_data = []
    with open(args.valid_path, 'r', encoding='utf-8') as f:
        for line in f:
            valid_data.append(line.strip())
    train_dataset = dataset(train_data,tokenizer)
    valid_dataset = dataset(valid_data,tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch_for_lm)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch_for_lm)
    
    config = {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "vocab_size": tokenizer.get_vocab_size(),
        "mode":"standard"
    }
    #initialize model
    model = LanguageModel(config).to(device)
    
    #loss fn
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0) 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_valid_loss = float('inf')
    os.makedirs(args.output_model_path, exist_ok=True)
    
    print("Starting training...")
    #training loop 
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            
            logits = model(input_ids, attention_mask)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = criterion(shift_logits.view(-1, config['vocab_size']), shift_labels.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
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
        valid_perplexity = calculate_perplexity(avg_valid_loss)
        print(
            f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f} "
            f"- Valid Loss: {avg_valid_loss:.4f} - Valid PPL: {valid_perplexity:.4f}"
        )
        #checkpointing
        if args.save_intermediate:
            checkpoint_path = os.path.join(args.output_model_path, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_valid_loss,
            }, checkpoint_path)
            print(f"Saved intermediate checkpoint to {checkpoint_path}")

        #save best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_model_path = os.path.join(args.output_model_path, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Saved new best model to {best_model_path}")
        


if __name__ == '__main__':
    import argparse
    #time
    start_time = time()

    parser = argparse.ArgumentParser(description='Train a model on the given dataset.')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the train dataset')
    parser.add_argument('--valid_path', type=str, required=True, help='Path to the valid dataset')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer')
    parser.add_argument('--output_model_path', type=str, default='checkpoints', help='Directory to save checkpoints')
    
    #hyperparameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--d_model', type=int, default=256, help='Dimension of the model')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of transformer layers')

    #checkpointing
    parser.add_argument('--save_intermediate', action='store_true', help='Flag to save intermediate optimizer/model states')
    
    
    args = parser.parse_args()
    main(args)
    end_time = time()
    elapsed = int(end_time - start_time)
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"Total training time: {hrs}h {mins}m {secs}s")
