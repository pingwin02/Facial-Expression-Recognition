#!/bin/bash
set -e

echo "======================================================"
echo "Starting all test runs with various configurations..."
echo "======================================================"

MODELS=("TransferModel" "ResNetModel")
INPUTS=("devemo_combined")
CLASSES=("binary" "all")

for model in "${MODELS[@]}"; do
    for input in "${INPUTS[@]}"; do
        for cls in "${CLASSES[@]}"; do
            echo ""
            echo "--- Model=$model Input=$input Class=$cls ---"
            python main.py --model "$model" --input "$input" --mode both --epochs 100 \
                --train-frame-selection uniform --test-frame-selection uniform \
                --num-frames 5 --class-split "$cls" --loop 10
        done
    done
done

echo "======================================================"
echo "All test runs completed successfully!"
echo "======================================================"
