#!/bin/bash

PARTA_DIR="parta"
ZIP_FILE="parta.zip"
GIT_FILE="$PARTA_DIR/git_commit.txt"

# a) Check if model.py exists
if [ ! -f "$PARTA_DIR/model.py" ]; then
    echo "Error: $PARTA_DIR/model.py not found."
    exit 1
fi

# b) Check if git_commit.txt exists and is non-empty
if [ ! -s "$GIT_FILE" ]; then
    echo "Error: $GIT_FILE does not exist or is empty."
    exit 1
fi

# c) Zip all .py files and git_commit.txt into parta.zip
# The zip will preserve the parta/ directory structure
echo "Creating $ZIP_FILE..."

zip -r "$ZIP_FILE" "$PARTA_DIR"/*.py "$GIT_FILE"

if [ $? -eq 0 ]; then
    echo "Zip created successfully: $ZIP_FILE"
else
    echo "Error creating zip file."
    exit 1
fi
