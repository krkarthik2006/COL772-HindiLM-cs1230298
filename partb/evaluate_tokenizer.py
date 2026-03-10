import os
import time
from .bpe_tokenizer import BPETokenizer
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


# Assert that encoding and then decoding gives back the original sentence.
def test_tokenizer_consistency(tokenizer, corpus, encoded_corpus):
    consistent = True
    inconsistent_sentences = 0
    for i in range(len(corpus)):
        original_sentence = corpus[i]
        encoded_tokens = encoded_corpus[i]
        reconstructed_sentence = tokenizer.decode(encoded_tokens)
        # assert original_sentence == reconstructed_sentence, f"Decoded text does not match original for sentence {i}!"
        if original_sentence != reconstructed_sentence:
            consistent = False
            inconsistent_sentences += 1

    return consistent, inconsistent_sentences


# Evaluate the tokenizer in terms of compression ratio
def calculate_compression_ratio(corpus, encoded_corpus):
    total_original_length = sum(len(sentence) for sentence in corpus)
    total_encoded_length = sum(len(tokens) for tokens in encoded_corpus) - len(corpus)  # Subtract 1 for each sentence to account for the end token
    compression_ratio = total_encoded_length / total_original_length
    return compression_ratio


# Compute out of vocabulary (OOV) rate
def calculate_oov_rate(encoded_corpus, unk_id):
    total_tokens = 0
    oov_tokens = 0
    for encoded_tokens in encoded_corpus:
        total_tokens += len(encoded_tokens)
        if unk_id is not None:
            oov_tokens += encoded_tokens.count(unk_id)
    oov_rate = oov_tokens / total_tokens if total_tokens > 0 else 0
    return oov_rate


# Analyze token frequency distribution and calculate a score based on the long tail of the distribution
def analyze_token_frequency(encoded_corpus, threshold=5):
    token_freq = {}
    for encoded_tokens in encoded_corpus:
        for token_id in encoded_tokens:
            token_freq[token_id] = token_freq.get(token_id, 0) + 1
    
    # Sort tokens by frequency
    sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Score based on how many tokens are in the long tail (e.g., tokens that appear less than a certain threshold)
    long_tail_tokens = [token for token, freq in sorted_tokens if freq < threshold]
    long_tail_score = len(long_tail_tokens) / len(token_freq) if token_freq else 0
    return long_tail_score


# Helper function for parallel encoding
def encode_sentence(sentence, tokenizer_path):
    """Encode a single sentence - used for multiprocessing"""
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    return tokenizer.encode(sentence)


# Helper function for parallel token frequency counting
def count_tokens_in_batch(encoded_batch):
    """Count token frequencies in a batch of encoded sentences"""
    token_freq = {}
    for encoded_tokens in encoded_batch:
        for token_id in encoded_tokens:
            token_freq[token_id] = token_freq.get(token_id, 0) + 1
    return token_freq


def main(args):
    corpus = []
    with open(args.input_corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(line.strip())
        # # First 1000 lines for evaluation
        # for _ in range(1000):
        #     line = f.readline()
        #     if not line:
        #         break
        #     corpus.append(line.strip())

    print(f"Loaded {len(corpus)} sentences from the evaluation dataset.")
    
    # Determine number of processes to use
    num_processes = args.num_processes if hasattr(args, 'num_processes') and args.num_processes else cpu_count()
    print(f"Using {num_processes} processes for parallel encoding.")
    
    # Load tokenizer (for non-parallel operations)
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer_path)
    print(f"Tokenizer loaded from {args.tokenizer_path}.")
    unk_id = tokenizer.get_unk_id()

    # Parallel encoding of all sentences
    print("Encoding corpus in parallel...")
    with Pool(processes=num_processes) as pool:
        encode_func = partial(encode_sentence, tokenizer_path=args.tokenizer_path)
        encoded_corpus = list(tqdm(
            pool.imap(encode_func, corpus, chunksize=max(1, len(corpus) // (num_processes * 4))),
            total=len(corpus),
            desc="Encoding"
        ))

    # Run all evaluations using pre-encoded corpus
    consistent, inconsistent_sentences = test_tokenizer_consistency(tokenizer, corpus, encoded_corpus)
    print(f"Tokenizer consistency test passed: {consistent}")
    if not consistent:
        print(f"Number of inconsistent sentences: {inconsistent_sentences}")
    
    compression_ratio = calculate_compression_ratio(corpus, encoded_corpus)
    print(f"Compression ratio: {compression_ratio:.4f}")

    oov_rate = calculate_oov_rate(encoded_corpus, unk_id)
    print(f"OOV rate: {oov_rate:.4f}")

    long_tail_score = analyze_token_frequency(encoded_corpus, threshold=5)
    print(f"Long-tail token score: {long_tail_score:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate a BPE tokenizer on the PMIndia dataset')
    parser.add_argument('--input_corpus_path', type=str, required=True, help='Path to the corpus text file for evaluation')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the trained tokenizer file')
    parser.add_argument('--num_processes', type=int, default=None, help='Number of parallel processes (default: CPU count)')
    args = parser.parse_args()

    main(args)
