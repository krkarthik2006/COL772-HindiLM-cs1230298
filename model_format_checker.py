import os
import re

import torch

from parta.model import LanguageModel, collate_fn
from partb.bpe_tokenizer import BPETokenizer

print("This script checks whether your model and tokenizer are compatible with the expected format for the assignment.")
print("Make sure to implement the load_model_and_tokenizer function to load your trained model and tokenizer from the checkpoint directory, and ensure that your model and tokenizer are compatible with the check_format function.")

def load_model_and_tokenizer(model_path, tokenizer_path):
    """
    CHANGE THIS FUNCTION TO LOAD YOUR TRAINED MODEL AND TOKENIZER FROM THE CHECKPOINT DIRECTORY.
    """
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)

    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    # Infer architecture where possible and keep the remaining values aligned
    # with defaults used in partc/train_model.py.
    d_model = state_dict["input_embeddings.weight"].shape[1]
    n_layers = len(
        {
            int(match.group(1))
            for key in state_dict
            if (match := re.match(r"transformer_blocks\.(\d+)\.", key))
        }
    )

    model_vocab_size = state_dict["input_embeddings.weight"].shape[0]
    tokenizer_vocab_size = tokenizer.get_vocab_size()
    if model_vocab_size != tokenizer_vocab_size:
        raise ValueError(
            f"Vocab mismatch: model expects {model_vocab_size} tokens, "
            f"but tokenizer has {tokenizer_vocab_size}. "
            "Please load the tokenizer used during model training."
        )

    config = {
        "d_model": d_model,
        "n_heads": 12 ,  # Assuming head size of 64 as in partc/train_model.py
        "n_layers": n_layers,
        "vocab_size": tokenizer_vocab_size,
        "dropout": 0.15,
        "attention_dropout": 0.15,
        "mode": "standard",
        "no_tie_embeddings": True,
    }

    model = LanguageModel(config)
    model.load_state_dict(state_dict)
    return model, tokenizer


def check_format(model, tokenizer, texts):
    """
    Hands-off. This function checks whether your model, tokenizer and collate_fn are compatible with the expected format for the assignment.
    YOU ARE NOT ALLOWED TO CHANGE THIS FUNCTION. Make sure your model and tokenizer are compatible with this function.
    """

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.to(device)
    model.eval()

    input_ids = [tokenizer.encode(text) for text in texts]
    assert all(isinstance(ids, list) for ids in input_ids), "Tokenizer encode method should return a list of token IDs for each input text."
    attention_masks = [[1] * len(ids) for ids in input_ids]
    input_ids = [torch.tensor(ids) for ids in input_ids]
    attention_masks = [torch.tensor(mask) for mask in attention_masks]
    batch = {'input_ids': input_ids, 'attention_mask': attention_masks}
    padded_batch = collate_fn(batch)
    assert 'input_ids' in padded_batch and 'attention_mask' in padded_batch, "Collate function should return a dictionary with 'input_ids' and 'attention_mask'."
    assert padded_batch['input_ids'].shape[0] == len(texts), "Batch size should match the number of input texts."
    assert padded_batch['input_ids'].shape[1] == max(len(ids) for ids in input_ids), "Sequence length should match the longest input text after padding."
    assert padded_batch['input_ids'].shape == padded_batch['attention_mask'].shape, "Input IDs and attention mask should have the same shape."
    assert isinstance(padded_batch['input_ids'], torch.Tensor), "Collate function should return input IDs as a PyTorch tensor."
    assert isinstance(padded_batch['attention_mask'], torch.Tensor), "Collate function should return attention mask as a PyTorch tensor."

    padded_input_ids = padded_batch['input_ids'].to(device)
    padded_attention_mask = padded_batch['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(padded_input_ids, padded_attention_mask)
        assert isinstance(logits, torch.Tensor), "Model should return logits as a PyTorch tensor."
        assert logits.shape[0] == len(texts), "Logits batch size should match the number of input texts."
        assert logits.shape[1] == max(len(ids) for ids in input_ids), "Logits sequence length should match the longest input text after padding."
        assert logits.shape[2] == tokenizer.get_vocab_size(), "Logits vocabulary size should match the tokenizer's vocabulary size."


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check model and tokenizer format")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint directory')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer checkpoint directory')
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.tokenizer_path)

    # Example texts to check format
    texts = [
        "यह एक परीक्षण है।",
        "मॉडल और टोकनाइज़र का प्रारूप सही होना चाहिए।",
        "यह सुनिश्चित करें कि सब कुछ अपेक्षित प्रारूप के अनुरूप है।"
    ]

    check_format(model, tokenizer, texts)
    print("Model and tokenizer format check passed successfully!")
