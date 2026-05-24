#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_SCRIPT="${SCRIPT_DIR}/run_tests.sh"

DETACHED=0
DEBUG=0
LOG_FILE="${SCRIPT_DIR}/out.log"
PID_FILE="${SCRIPT_DIR}/out.pid"

cleanup_pid_file() {
    if [[ -f "${PID_FILE}" ]] && [[ "$(cat "${PID_FILE}")" == "$$" ]]; then
        rm -f "${PID_FILE}"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --detached)
            DETACHED=1
            shift
            ;;
        --debug)
            DEBUG=1
            shift
            ;;
        --help|-h)
            echo "Usage: ./run_tests.sh [--detached] [--debug]"
            echo "  --detached  Run the test suite in the background and save PID to ${PID_FILE}"
            echo "  --debug     Override the suite to 1 loop and 10 epochs"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: ./run_tests.sh [--detached] [--debug]" >&2
            exit 1
            ;;
    esac
done

if [[ ${DEBUG} -eq 1 ]]; then
    EPOCHS=10
    LOOPS=1
else
    EPOCHS=100
    LOOPS=10
fi

if [[ ${DETACHED} -eq 1 ]]; then
    nohup "${SELF_SCRIPT}" $([[ ${DEBUG} -eq 1 ]] && printf '%s' '--debug') &> "${LOG_FILE}" &
    echo $! > "${PID_FILE}"
    echo "Started run_tests.sh in background. PID saved to ${PID_FILE}, logs: ${LOG_FILE}"
    exit 0
fi

cd "${SCRIPT_DIR}"

if [[ -f "${PID_FILE}" ]] && [[ "$(cat "${PID_FILE}")" == "$$" ]]; then
    trap cleanup_pid_file EXIT
fi

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
            echo "--- Model=$model Input=$input Class=$cls Epochs=$EPOCHS Loops=$LOOPS ---"
            python main.py --model "$model" --input "$input" --mode both --epochs "$EPOCHS" \
                --train-frame-selection uniform --test-frame-selection uniform \
                --num-frames 5 --class-split "$cls" --loop "$LOOPS"
        done
    done
done

echo "--- Average Model Performance ---"
python misc/average_runs.py -n "$LOOPS"

echo "======================================================"
echo "All test runs completed successfully!"
echo "======================================================"
