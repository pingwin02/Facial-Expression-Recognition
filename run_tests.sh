#!/bin/bash
set -e

echo "==============================================="

# devemo_combined (input index 4), uniform/uniform, loop 10, binary and all classes

./train_eval.sh -m 0 -i 4 -t 0 -T 0 -f 5 -l 10 -c 0

./train_eval.sh -m 0 -i 4 -t 0 -T 0 -f 5 -l 10 -c 1

./train_eval.sh -m 1 -i 4 -t 0 -T 0 -f 5 -l 10 -c 0

./train_eval.sh -m 1 -i 4 -t 0 -T 0 -f 5 -l 10 -c 1

echo ""
echo "==============================================="
echo "All test runs completed successfully!"
