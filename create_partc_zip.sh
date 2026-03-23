#!/bin/bash

PARTA_DIR="parta"
PARTB_DIR="partb"
PARTC_DIR="partc"
ZIP_FILE="partc.zip"
PARTC_GIT_FILE="$PARTC_DIR/git_commit.txt"
PARTC_CHECKPOINT_FILE="$PARTC_DIR/checkpoint.txt"
FORMAT_CHECKER_FILE="model_format_checker.py"

if [ ! -f "$PARTA_DIR/model.py" ]; then
    echo "Error: $PARTA_DIR/model.py not found."
    exit 1
fi

if [ ! -f "$PARTB_DIR/bpe_tokenizer.py" ]; then
    echo "Error: $PARTB_DIR/bpe_tokenizer.py not found."
    exit 1
fi

if [ ! -f "$PARTC_DIR/train_model.py" ]; then
    echo "Error: $PARTC_DIR/train_model.py not found."
    exit 1
fi

if [ ! -s "$PARTC_GIT_FILE" ]; then
    echo "Error: $PARTC_GIT_FILE does not exist or is empty."
    exit 1
fi

if [ ! -s "$PARTC_CHECKPOINT_FILE" ]; then
    echo "Error: $PARTC_CHECKPOINT_FILE does not exist or is empty."
    exit 1
fi

if [ ! -f "$FORMAT_CHECKER_FILE" ]; then
    echo "Error: $FORMAT_CHECKER_FILE not found."
    exit 1
fi

echo "Creating $ZIP_FILE..."

zip -r "$ZIP_FILE" \
    "$PARTA_DIR" \
    "$PARTB_DIR" \
    "$PARTC_DIR" \
    "$FORMAT_CHECKER_FILE" \
    -i "*.py" "$PARTC_GIT_FILE" "$PARTC_CHECKPOINT_FILE" "$FORMAT_CHECKER_FILE" \
    -x "*.ipynb_checkpoints*" -x "*__pycache__*" -x "*.pt" -x "*.pth"

if [ $? -eq 0 ]; then
    echo "Zip created successfully: $ZIP_FILE"
else
    echo "Error creating zip file."
    exit 1
fi
