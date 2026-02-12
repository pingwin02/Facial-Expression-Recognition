#!/bin/bash

EPOCHS=100
MODEL_INDEX_ARG=""
INPUT_INDEX_ARG=""

usage() {
    echo "Usage: $0 [-e epochs] [-m model_index] [-i input_index]"
    echo "  -m and -i accept numeric menu indices."
}

while getopts "e:m:i:" opt; do
    case $opt in
        e) EPOCHS=$OPTARG ;;
        m) MODEL_INDEX_ARG=$OPTARG ;;
        i) INPUT_INDEX_ARG=$OPTARG ;;
        *) usage; exit 1 ;;
    esac
done

MODELS_DIR="./models"
INPUT_DIR="./input"

if [ ! -d "$MODELS_DIR" ] || [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Required directories do not exist."
    exit 1
fi

REAL_MODELS=()
for f in "$MODELS_DIR"/*.py; do
    [ -e "$f" ] && REAL_MODELS+=("$(basename "$f" .py)")
done

if [ ${#REAL_MODELS[@]} -eq 0 ]; then
    echo "Error: No model files found in $MODELS_DIR."
    exit 1
fi

MENU_OPTIONS=("${REAL_MODELS[@]}" "All models")
echo "Available models:"
for i in "${!MENU_OPTIONS[@]}"; do echo "$i) ${MENU_OPTIONS[$i]}"; done

if [ -n "$MODEL_INDEX_ARG" ]; then
    MODEL_INDEX=$MODEL_INDEX_ARG
    echo "Model selection from args: $MODEL_INDEX"
else
    read -p "Select a model by number: " MODEL_INDEX
fi

if ! [[ $MODEL_INDEX =~ ^[0-9]+$ ]] || [[ $MODEL_INDEX -lt 0 || $MODEL_INDEX -ge ${#MENU_OPTIONS[@]} ]]; then
    echo "Invalid selection."
    exit 1
fi

SELECTED_OPTION=${MENU_OPTIONS[$MODEL_INDEX]}
MODELS_TO_RUN=("${REAL_MODELS[@]}")
[[ "$SELECTED_OPTION" != "All models" ]] && MODELS_TO_RUN=("$SELECTED_OPTION")

REAL_INPUTS=()
for DIR in "$INPUT_DIR"/*/; do
    [ -d "$DIR" ] || continue
    INPUT_NAME=$(basename "$DIR")
    [[ "$INPUT_NAME" == ".cache" || "$INPUT_NAME" == "__pycache__" ]] && continue
    REAL_INPUTS+=("$INPUT_NAME")
done

if [ ${#REAL_INPUTS[@]} -eq 0 ]; then
    echo "Error: No input directories found in $INPUT_DIR."
    exit 1
fi

INPUT_OPTIONS=("${REAL_INPUTS[@]}" "All inputs")
echo "Available inputs:"
for i in "${!INPUT_OPTIONS[@]}"; do echo "$i) ${INPUT_OPTIONS[$i]}"; done

if [ -n "$INPUT_INDEX_ARG" ]; then
    INPUT_INDEX=$INPUT_INDEX_ARG
    echo "Input selection from args: $INPUT_INDEX"
else
    read -p "Select an input by number: " INPUT_INDEX
fi

if ! [[ $INPUT_INDEX =~ ^[0-9]+$ ]] || [[ $INPUT_INDEX -lt 0 || $INPUT_INDEX -ge ${#INPUT_OPTIONS[@]} ]]; then
    echo "Invalid selection."
    exit 1
fi

SELECTED_INPUT_OPTION=${INPUT_OPTIONS[$INPUT_INDEX]}
INPUTS_TO_RUN=("${REAL_INPUTS[@]}")
[[ "$SELECTED_INPUT_OPTION" != "All inputs" ]] && INPUTS_TO_RUN=("$SELECTED_INPUT_OPTION")

for MODE in train eval; do
    for INPUT_NAME in "${INPUTS_TO_RUN[@]}"; do
        for MODEL in "${MODELS_TO_RUN[@]}"; do
            python main.py --input "$INPUT_NAME" --mode "$MODE" --epochs "$EPOCHS" --model "$MODEL"
        done
    done
done
