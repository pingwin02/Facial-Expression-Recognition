#!/bin/bash

# Test runs configuration:
# Model: 0 (TransferModel)
# Inputs: 0 (devemo+), 1 (devemo)
# Training selections: 0 (uniform), 2 (random)
# Test selections:
#   - test 0 (same as training) for each training
#   - PLUS: when train 0 → test 3 (random), when train 2 → test 1 (uniform)
# Frames: 5
# Loops: 10
# Class split: 0 (binary)

set -e

echo "Starting test runs (8 total)..."
echo "==============================================="

# Input 0 (devemo+), Training 0 (uniform)
echo "Run 1/8: Model 0, Input 0, Train 0 (uniform), Test 0 (same=uniform), Frames 5, Loops 10"
./train_eval.sh -m 0 -i 0 -t 0 -T 0 -f 5 -l 10 -c 0

echo ""
echo "Run 2/8: Model 0, Input 0, Train 0 (uniform), Test 3 (random), Frames 5, Loops 10"
./train_eval.sh -m 0 -i 0 -t 0 -T 3 -f 5 -l 10 -c 0

# Input 0 (devemo+), Training 2 (random)
echo ""
echo "Run 3/8: Model 0, Input 0, Train 2 (random), Test 0 (same=random), Frames 5, Loops 10"
./train_eval.sh -m 0 -i 0 -t 2 -T 0 -f 5 -l 10 -c 0

echo ""
echo "Run 4/8: Model 0, Input 0, Train 2 (random), Test 1 (uniform), Frames 5, Loops 10"
./train_eval.sh -m 0 -i 0 -t 2 -T 1 -f 5 -l 10 -c 0

# Input 1 (devemo), Training 0 (uniform)
echo ""
echo "Run 5/8: Model 0, Input 1, Train 0 (uniform), Test 0 (same=uniform), Frames 5, Loops 10"
./train_eval.sh -m 0 -i 1 -t 0 -T 0 -f 5 -l 10 -c 0

echo ""
echo "Run 6/8: Model 0, Input 1, Train 0 (uniform), Test 3 (random), Frames 5, Loops 10"
./train_eval.sh -m 0 -i 1 -t 0 -T 3 -f 5 -l 10 -c 0

# Input 1 (devemo), Training 2 (random)
echo ""
echo "Run 7/8: Model 0, Input 1, Train 2 (random), Test 0 (same=random), Frames 5, Loops 10"
./train_eval.sh -m 0 -i 1 -t 2 -T 0 -f 5 -l 10 -c 0

echo ""
echo "Run 8/8: Model 0, Input 1, Train 2 (random), Test 1 (uniform), Frames 5, Loops 10"
./train_eval.sh -m 0 -i 1 -t 2 -T 1 -f 5 -l 10 -c 0

echo ""
echo "==============================================="
echo "All test runs completed successfully!"
