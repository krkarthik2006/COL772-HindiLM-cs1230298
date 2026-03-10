# COL772-HindiLLM: Assignment on Building a Hindi Language Model

Welcome, students, to the COL772 HindiLLM assignment repository! This assignment is designed to guide you through the process of building and training components of a Large Language Model (LLM) specifically for the Hindi language. You will be working on different stages, from data preparation and tokenization to model training.

## Table of Contents
1.  [Project Structure](#project-structure)
2.  [Setup Instructions](#setup-instructions)
3.  [Running the Assignment Parts](#running-the-assignment-parts)
    *   [Part A: Initial Model](#part-a-data-preparation-and-initial-model)
    *   [Part B: BPE Tokenizer Implementation](#part-b-bpe-tokenizer-implementation)
    *   [Part C: Training the Language Model](#part-c-training-the-language-model)
4.  [Important Notes](#important-notes)

## Project Structure

This repository is organized into several directories, each corresponding to a specific part of the assignment or containing shared resources.

```
.
├── __init__.py
├── model_format_checker.py  # Utility to check model output format
├── run_parta.sh             # Script to run Part A
├── run_partb.sh             # Script to run Part B
├── run_partc.sh             # Script to run Part C
├── data/
│   ├── tokenizer_corpus.txt # Corpus for tokenizer training
│   ├── train.txt            # Training data for the language model
│   └── valid.txt            # Validation data for the language model
├── parta/
│   ├── __init__.py
│   ├── check.py             # Script to check Part A implementation
│   └── model.py             # Your implementation for Part A goes here
 specific to Part A
├── partb/
│   ├── __init__.py
│   ├── bpe_tokenizer.py     # Your implementation for BPE tokenizer goes here
│   └── train_tokenizer.py   # Script to train the BPE tokenizer
└── partc/
    ├── __init__.py
    ├── train_model.py       # Your implementation for model training goes here
    └── utils.py             # Utility functions for Part C
```

## Setup Instructions

Read the assignment document carefully to implement this code into your private repository.
**You are responsible for protecting your code.**

Once you have setup your private repository, fill up [this form](https://forms.office.com/r/MmSrhXrRn1).



## Running the Assignment Parts

Each part of the assignment can be run using a dedicated shell script. These scripts are designed to help you execute your code and test your implementations.

### Part A: Data Preparation and Initial Model

In this part, you will focus on implementing a basic model and data collating logic.

To run Part A:
```bash
bash run_parta.sh
```
You will need to implement the necessary logic in `parta/model.py` and ensure `parta/check.py` passes.

### Part B: BPE Tokenizer Implementation

Part B involves implementing a Byte Pair Encoding (BPE) tokenizer from scratch or using a library.

To run Part B:
```bash
bash run_partb.sh
```
Your BPE tokenizer implementation will go into `partb/bpe_tokenizer.py`. The `partb/train_tokenizer.py` script will use your implementation to train a tokenizer on `data/tokenizer_corpus.txt`.

### Part C: Training the Language Model

In Part C, you will integrate your tokenizer and implement the training loop for a language model.

To run Part C:
```bash
bash /run_partc.sh --train-tokenizer|--train-model
```
The core training logic will reside in `partc/train_model.py`.

## Important Notes

*   **Read the problem statement carefully:** Before starting any part, make sure you thoroughly understand the requirements and expectations.
*   **Incremental Development:** Work on one part at a time. Test your code frequently.
*   **Use the provided data:** The `data/` directory contains the necessary text files for training your tokenizer and language model.
*   **Check scripts:** The `run_part*.sh` scripts and `check.py` files are crucial for verifying your implementation. Make sure your code runs correctly through these scripts.
*   **Code Style:** Maintain clean, readable, and well-commented code.

## Queries and Forum

All queries and discussion related to the assignment must be done on Piazza. Any direct messages/emails to TAs will not be entertained.

----

Good luck with the assignment! We hope you find this a rewarding learning experience.


# Changlog

**How to run `evaluate_tokenizer.py` file?**

Run the following command form the ROOT of the package. Do not run from `partb/` directory.
```
python -m partb.evaluate_tokenizer --input_corpus_path ./data/tokenizer_corpus.txt --tokenizer_path <path to your tokenizer>
```
