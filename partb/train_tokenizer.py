import os
import time
from .bpe_tokenizer import BPETokenizer


def main(args):
    corpus = []
    with open(args.input_corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(line.strip())

    if args.train_path is not None:
        print(f"Loading training data from {args.train_path}...")
        with open(args.train_path, 'r', encoding='utf-8') as f:
            for line in f:
                corpus.append(line.strip())

    print(f"Loaded {len(corpus)} sentences from the dataset.")
    tokenizer = BPETokenizer(args.vocab_size)
    st = time.time()
    tokenizer.train(corpus)
    en = time.time()

    print(f"Training completed in {en - st:.2f} seconds.")
    # 1. Extract just the folder path (e.g., "./partb/final_tokenizer")
    output_dir = os.path.dirname(args.output_tokenizer_path)
    
    # 2. Safely create only the folder (if a folder path actually exists)
    if output_dir:  
        os.makedirs(output_dir, exist_ok=True)

    # 3. Now save the file!
    tokenizer.save(args.output_tokenizer_path)

    print(f"Sentence: {corpus[0]}")
    print(f"Tokens: {tokenizer.encode(corpus[0])}")
    print(f"Reconstructed: {tokenizer.decode(tokenizer.encode(corpus[0]))}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a BPE tokenizer on the provided dataset')
    parser.add_argument('--input_corpus_path', type=str, required=True, help='Path to the input corpus text file')
    parser.add_argument('--train_path', type=str, default=None, required=False, help='Path to the training data text file. Only used in Part C of the assignment.')
    parser.add_argument('--vocab_size', type=int, default=1000, help='Vocabulary size for the BPE tokenizer')
    parser.add_argument('--output_tokenizer_path', type=str, required=True, help='Path to save the trained tokenizer')
    args = parser.parse_args()

    main(args)
