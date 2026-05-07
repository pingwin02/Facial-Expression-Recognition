#!/bin/bash

EPOCHS=100
MODEL_INDEX_ARG=""
INPUT_INDEX_ARG=""
MODE_ARG=""
LOOP_COUNT=1
TRAIN_FRAME_INDEX_ARG=""
TEST_FRAME_INDEX_ARG=""
NUM_FRAMES_ARG=""
CLASS_SPLIT_INDEX_ARG=""

usage() {
    echo "Usage: $0 [-e epochs] [-m model_index] [-i input_index] [-M mode] [-l loop_count]"
    echo "          [-t train_frame_index] [-T test_frame_index] [-f num_frames] [-c class_split_index]"
    echo "  -m and -i accept numeric menu indices."
    echo "  -M accepts: train, eval, or both (default: both)"
    echo "  -l accepts number of full run loops (default: 1)"
    echo "  -t accepts: 0=uniform, 1=transformer, 2=random, 3=manual_uniform, 4=manual_random, 5=manual_transformer"
    echo "  -T accepts: 0=same_as_train, 1=uniform, 2=transformer, 3=random, 4=manual_uniform, 5=manual_random, 6=manual_transformer"
    echo "  -f accepts number of frames per video (default: 5)"
    echo "  -c accepts: 0=binary, 1=all"
}

while getopts "e:m:i:M:l:t:T:f:c:" opt; do
    case $opt in
        e) EPOCHS=$OPTARG ;;
        m) MODEL_INDEX_ARG=$OPTARG ;;
        i) INPUT_INDEX_ARG=$OPTARG ;;
        M) MODE_ARG=$OPTARG ;;
        l) LOOP_COUNT=$OPTARG ;;
        t) TRAIN_FRAME_INDEX_ARG=$OPTARG ;;
        T) TEST_FRAME_INDEX_ARG=$OPTARG ;;
        f) NUM_FRAMES_ARG=$OPTARG ;;
        c) CLASS_SPLIT_INDEX_ARG=$OPTARG ;;
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

# Frame selection and class split options (only for devemo/devemo+)
TRAIN_FRAME_SEL=""
TEST_FRAME_SEL=""
NUM_FRAMES=""
CLASS_SPLIT=""

IS_DEVEMO=false
for INPUT_NAME in "${INPUTS_TO_RUN[@]}"; do
    if [[ "$INPUT_NAME" == "devemo" || "$INPUT_NAME" == "devemo+" ]]; then
        IS_DEVEMO=true
        break
    fi
done

if [ "$IS_DEVEMO" = true ]; then
    echo ""
    echo "=== Frame Selection Options ==="
    echo "Training frame selection method:"
    echo "0) uniform (default - evenly spaced)"
    echo "1) transformer (ViT-based attention)"
    echo "2) random"
    echo "3) manual + uniform (pick frames manually, rest uniform)"
    echo "4) manual + random (pick frames manually, rest random)"
    echo "5) manual + transformer (pick frames manually, rest transformer)"
    if [ -n "$TRAIN_FRAME_INDEX_ARG" ]; then
        TRAIN_FRAME_IDX=$TRAIN_FRAME_INDEX_ARG
        echo "Training frame selection from args: $TRAIN_FRAME_IDX"
    else
        read -p "Select training frame method [0]: " TRAIN_FRAME_IDX
    fi
    TRAIN_FRAME_IDX=${TRAIN_FRAME_IDX:-0}

    case $TRAIN_FRAME_IDX in
        0) TRAIN_FRAME_SEL="uniform" ;;
        1) TRAIN_FRAME_SEL="transformer" ;;
        2) TRAIN_FRAME_SEL="random" ;;
        3) TRAIN_FRAME_SEL="manual_uniform" ;;
        4) TRAIN_FRAME_SEL="manual_random" ;;
        5) TRAIN_FRAME_SEL="manual_transformer" ;;
        *) echo "Invalid selection, using uniform."; TRAIN_FRAME_SEL="uniform" ;;
    esac

    echo ""
    echo "Test frame selection method (default: same as training = $TRAIN_FRAME_SEL):"
    echo "0) same as training"
    echo "1) uniform"
    echo "2) transformer"
    echo "3) random"
    echo "4) manual + uniform"
    echo "5) manual + random"
    echo "6) manual + transformer"
    if [ -n "$TEST_FRAME_INDEX_ARG" ]; then
        TEST_FRAME_IDX=$TEST_FRAME_INDEX_ARG
        echo "Test frame selection from args: $TEST_FRAME_IDX"
    else
        read -p "Select test frame method [0]: " TEST_FRAME_IDX
    fi
    TEST_FRAME_IDX=${TEST_FRAME_IDX:-0}

    case $TEST_FRAME_IDX in
        0) TEST_FRAME_SEL="$TRAIN_FRAME_SEL" ;;
        1) TEST_FRAME_SEL="uniform" ;;
        2) TEST_FRAME_SEL="transformer" ;;
        3) TEST_FRAME_SEL="random" ;;
        4) TEST_FRAME_SEL="manual_uniform" ;;
        5) TEST_FRAME_SEL="manual_random" ;;
        6) TEST_FRAME_SEL="manual_transformer" ;;
        *) echo "Invalid selection, using same as training."; TEST_FRAME_SEL="$TRAIN_FRAME_SEL" ;;
    esac

    echo ""
    if [ -n "$NUM_FRAMES_ARG" ]; then
        NUM_FRAMES=$NUM_FRAMES_ARG
        echo "Number of frames from args: $NUM_FRAMES"
    else
        read -p "Number of frames to select per video [5]: " NUM_FRAMES_INPUT
        NUM_FRAMES=${NUM_FRAMES_INPUT:-5}
    fi

    if ! [[ $NUM_FRAMES =~ ^[0-9]+$ ]] || [[ $NUM_FRAMES -lt 1 ]]; then
        echo "Error: Invalid number of frames '$NUM_FRAMES'. Use an integer >= 1."
        exit 1
    fi

    echo ""
    echo "Class split:"
    echo "0) binary (negative / others) - default"
    echo "1) all (all original classes)"
    if [ -n "$CLASS_SPLIT_INDEX_ARG" ]; then
        CLASS_SPLIT_IDX=$CLASS_SPLIT_INDEX_ARG
        echo "Class split selection from args: $CLASS_SPLIT_IDX"
    else
        read -p "Select class split [0]: " CLASS_SPLIT_IDX
    fi
    CLASS_SPLIT_IDX=${CLASS_SPLIT_IDX:-0}

    case $CLASS_SPLIT_IDX in
        0) CLASS_SPLIT="binary" ;;
        1) CLASS_SPLIT="all" ;;
        *) echo "Invalid selection, using binary."; CLASS_SPLIT="binary" ;;
    esac

    echo ""
    echo "Selected: train=$TRAIN_FRAME_SEL, test=$TEST_FRAME_SEL, frames=$NUM_FRAMES, classes=$CLASS_SPLIT"
elif [ -n "${TRAIN_FRAME_INDEX_ARG}${TEST_FRAME_INDEX_ARG}${NUM_FRAMES_ARG}${CLASS_SPLIT_INDEX_ARG}" ]; then
    echo "Ignoring frame-selection arguments because selected inputs do not include devemo/devemo+."
fi

if [ -z "$MODE_ARG" ] || [ "$MODE_ARG" == "both" ]; then
    MODES_TO_RUN=("train" "eval")
elif [ "$MODE_ARG" == "train" ] || [ "$MODE_ARG" == "eval" ]; then
    MODES_TO_RUN=("$MODE_ARG")
else
    echo "Error: Invalid mode '$MODE_ARG'. Use 'train', 'eval', or 'both'."
    exit 1
fi

if ! [[ $LOOP_COUNT =~ ^[0-9]+$ ]] || [[ $LOOP_COUNT -lt 1 ]]; then
    echo "Error: Invalid loop count '$LOOP_COUNT'. Use an integer >= 1."
    exit 1
fi

for ((LOOP_INDEX=1; LOOP_INDEX<=LOOP_COUNT; LOOP_INDEX++)); do
    echo "Starting loop $LOOP_INDEX/$LOOP_COUNT"

    for MODE in "${MODES_TO_RUN[@]}"; do
        for INPUT_NAME in "${INPUTS_TO_RUN[@]}"; do
            for MODEL in "${MODELS_TO_RUN[@]}"; do
                CMD=(python -u main.py --input "$INPUT_NAME" --mode "$MODE" --epochs "$EPOCHS" --model "$MODEL")

                # Add frame selection args only for devemo/devemo+
                if [[ "$INPUT_NAME" == "devemo" || "$INPUT_NAME" == "devemo+" ]]; then
                    if [ -n "$TRAIN_FRAME_SEL" ]; then
                        CMD+=(--train-frame-selection "$TRAIN_FRAME_SEL")
                    fi
                    if [ -n "$TEST_FRAME_SEL" ]; then
                        CMD+=(--test-frame-selection "$TEST_FRAME_SEL")
                    fi
                    if [ -n "$NUM_FRAMES" ]; then
                        CMD+=(--num-frames "$NUM_FRAMES")
                    fi
                    if [ -n "$CLASS_SPLIT" ]; then
                        CMD+=(--class-split "$CLASS_SPLIT")
                    fi
                fi

                "${CMD[@]}"
                if [ $? -ne 0 ]; then
                    echo "Error: Command failed in loop $LOOP_INDEX/$LOOP_COUNT. Exiting."
                    exit 1
                fi
            done
        done
    done
done

if [ -f ~/.netrc ]; then
    rm ~/.netrc
    echo "Logged out from Weights & Biases (.netrc removed)"
fi