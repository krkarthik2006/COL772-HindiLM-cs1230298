#!/bin/bash

PARTB_DIR="partb"
ZIP_FILE="partb.zip"
GIT_FILE="$PARTB_DIR/git_commit.txt"

# a) Check if bpe_tokenizer.py exists
if [ ! -f "$PARTB_DIR/bpe_tokenizer.py" ]; then
    echo "Error: $PARTB_DIR/bpe_tokenizer.py not found."
    exit 1
fi

# b) Check if git_commit.txt exists and is non-empty
if [ ! -s "$GIT_FILE" ]; then
    echo "Error: $GIT_FILE does not exist or is empty."
    exit 1
fi

# c) Zip all .py files, any .pt files (recursively), and git_commit.txt into partb.zip
# The zip will preserve the partb/ directory structure
echo "Creating $ZIP_FILE..."

zip -r "$ZIP_FILE" "$PARTB_DIR" -i "*.py" "$GIT_FILE" -x "*.ipynb_checkpoints*" -x "*__pycache__" -x "*.pt"


if [ $? -eq 0 ]; then
    echo "Zip created successfully: $ZIP_FILE"
else
    echo "Error creating zip file."
    exit 1
fi
