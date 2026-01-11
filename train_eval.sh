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

if [ ! -d "$MODELS_DIR" ] || [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Required directories do not exist."
    exit 1
fi

REAL_MODELS=()
for f in "$MODELS_DIR"/*.py; do
    [ -e "$f" ] && REAL_MODELS+=("$(basename "$f" .py)")
done

MENU_OPTIONS=("${REAL_MODELS[@]}" "All models")
echo "Available models:"
for i in "${!MENU_OPTIONS[@]}"; do echo "$i) ${MENU_OPTIONS[$i]}"; done

read -p "Select a model by number: " MODEL_INDEX
if [[ $MODEL_INDEX -lt 0 || $MODEL_INDEX -ge ${#MENU_OPTIONS[@]} ]]; then
    echo "Invalid selection."
    exit 1
fi

SELECTED_OPTION=${MENU_OPTIONS[$MODEL_INDEX]}
MODELS_TO_RUN=("${REAL_MODELS[@]}")
[[ "$SELECTED_OPTION" != "All models" ]] && MODELS_TO_RUN=("$SELECTED_OPTION")

for MODE in train eval; do
    for DIR in "$INPUT_DIR"/*/; do
        [[ -d $DIR && $(basename "$DIR") != ".cache" && $(basename "$DIR") != "__pycache__" ]] || continue
        for MODEL in "${MODELS_TO_RUN[@]}"; do
            python main.py --input "$(basename "$DIR")" --mode "$MODE" --epochs "$EPOCHS" --model "$MODEL"
        done
    done
done
