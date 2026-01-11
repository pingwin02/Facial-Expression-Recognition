#!/bin/bash

EPOCHS=100

while getopts "e:" opt; do
    case $opt in
        e) EPOCHS=$OPTARG ;;
        *) echo "Usage: $0 [-e epochs]"; exit 1 ;;
    esac
done

MODELS_DIR="./models"
INPUT_DIR="./input"

REAL_MODELS=()
if [ -d "$MODELS_DIR" ]; then
    for f in "$MODELS_DIR"/*.py; do
        [ -e "$f" ] || continue
        filename=$(basename "$f" .py)
        REAL_MODELS+=("$filename")
    done
else
    echo "Error: Directory $MODELS_DIR does not exist."
    exit 1
fi

MENU_OPTIONS=("${REAL_MODELS[@]}" "All models")

echo "Available models:"
for i in "${!MENU_OPTIONS[@]}"; do
    echo "$i) ${MENU_OPTIONS[$i]}"
done

read -p "Select a model by number: " MODEL_INDEX

if [[ $MODEL_INDEX -lt 0 || $MODEL_INDEX -ge ${#MENU_OPTIONS[@]} ]]; then
    echo "Invalid selection."
    exit 1
fi

SELECTED_OPTION=${MENU_OPTIONS[$MODEL_INDEX]}

if [[ "$SELECTED_OPTION" == "All models" ]]; then
    MODELS_TO_RUN=("${REAL_MODELS[@]}")
else
    MODELS_TO_RUN=("$SELECTED_OPTION")
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory $INPUT_DIR does not exist."
    exit 1
fi

for DIR in "$INPUT_DIR"/*/; do
    if [[ -d $DIR ]]; then
        DIR_NAME=$(basename "$DIR")

        if [[ "$DIR_NAME" == ".cache" || "$DIR_NAME" == "__pycache__" ]]; then
            continue
        fi

        echo "Processing input: $DIR_NAME"

        for MODEL in "${MODELS_TO_RUN[@]}"; do
            python main.py --input "$DIR_NAME" --mode train --epochs "$EPOCHS" --model "$MODEL"
            python main.py --input "$DIR_NAME" --mode eval --model "$MODEL"
        done
    fi
done