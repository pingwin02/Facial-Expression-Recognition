#!/bin/bash
set -e

echo "==============================================="

./train_eval.sh -m 0 -i 0 -t 0 -T 0 -f 5 -l 5 -c 0

./train_eval.sh -m 0 -i 0 -t 0 -T 3 -f 5 -l 5 -c 0

./train_eval.sh -m 0 -i 0 -t 2 -T 0 -f 5 -l 5 -c 0

./train_eval.sh -m 0 -i 0 -t 2 -T 1 -f 5 -l 5 -c 0

./train_eval.sh -m 0 -i 1 -t 0 -T 0 -f 5 -l 5 -c 0

./train_eval.sh -m 0 -i 1 -t 0 -T 3 -f 5 -l 5 -c 0

./train_eval.sh -m 0 -i 1 -t 2 -T 0 -f 5 -l 5 -c 0

./train_eval.sh -m 0 -i 1 -t 2 -T 1 -f 5 -l 5 -c 0

echo ""
echo "==============================================="
echo "All test runs completed successfully!"
