#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE}")" && pwd)"
SELF_SCRIPT="${SCRIPT_DIR}/run_tests.sh"

DETACHED=0
DEBUG=0
CACHE_ONLY=0
LOG_FILE="${SCRIPT_DIR}/out.log"
PID_FILE="${SCRIPT_DIR}/out.pid"
DEFAULT_NUM_FRAMES=5

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
        --cache-only)
            CACHE_ONLY=1
            shift
            ;;
        --help|-h)
            echo "Usage: ./run_tests.sh [--detached] [--debug] [--cache-only]"
            echo "  --detached    Run the test suite in the background and save PID to ${PID_FILE}"
            echo "  --debug       Override the suite to 1 loop and 10 epochs"
            echo "  --cache-only  Run main.py with --cache-only"
            echo "Environment variables:"
            echo "  NUM_FRAMES             Override --num-frames for the run."
            echo "                         In --cache-only mode VEATIC defaults to 300 if unset; otherwise 5."
            echo "  DATASET_CACHE_TO_BUILD Dataset name used with --cache-only (e.g. veatic)."
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: ./run_tests.sh [--detached] [--debug] [--cache-only]" >&2
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
    DETACHED_CMD=("${SELF_SCRIPT}")
    if [[ ${DEBUG} -eq 1 ]]; then
        DETACHED_CMD+=(--debug)
    fi
    if [[ ${CACHE_ONLY} -eq 1 ]]; then
        DETACHED_CMD+=(--cache-only)
    fi

    nohup setsid "${DETACHED_CMD[@]}" &> "${LOG_FILE}" &

    echo $! > "${PID_FILE}"
    echo "Started run_tests.sh in background. PID saved to ${PID_FILE}, logs: ${LOG_FILE}"
    exit 0
fi

cd "${SCRIPT_DIR}"

if [[ -f "${PID_FILE}" ]] && [[ "$(cat "${PID_FILE}")" == "$$" ]]; then
    trap cleanup_pid_file EXIT
fi

if [[ -n "${NUM_FRAMES:-}" ]]; then
    if ! [[ "${NUM_FRAMES}" =~ ^[0-9]+$ ]] || [[ "${NUM_FRAMES}" -lt 1 ]]; then
        echo "NUM_FRAMES must be an integer >= 1." >&2
        exit 1
    fi
fi

if [[ ${CACHE_ONLY} -eq 1 ]]; then
    if [[ -z "${DATASET_CACHE_TO_BUILD:-}" ]]; then
        echo "DATASET_CACHE_TO_BUILD must be set when using --cache-only." >&2
        exit 1
    fi

    CACHE_NUM_FRAMES="${NUM_FRAMES:-}"
    if [[ -z "${CACHE_NUM_FRAMES}" ]]; then
        if [[ "${DATASET_CACHE_TO_BUILD}" == "veatic" ]]; then
            CACHE_NUM_FRAMES=300
        else
            CACHE_NUM_FRAMES=${DEFAULT_NUM_FRAMES}
        fi
    fi

    echo "======================================================"
    echo "Building cache..."
    echo "======================================================"
    echo "Dataset: ${DATASET_CACHE_TO_BUILD}"
    echo "Frames per video: ${CACHE_NUM_FRAMES}"
    python main.py --model 0 --input "${DATASET_CACHE_TO_BUILD}" --mode both \
        --epochs 1 --train-frame-selection uniform --test-frame-selection uniform \
        --num-frames "${CACHE_NUM_FRAMES}" --class-split "binary" --loop 1 --cache-only
    echo "======================================================"
    echo "Cache built successfully!"
    echo "======================================================"
    exit 0
fi

RUN_NUM_FRAMES="${NUM_FRAMES:-${DEFAULT_NUM_FRAMES}}"

echo "======================================================"
echo "Starting all test runs with various configurations..."
echo "======================================================"

MODELS=("TransferModel" "ResNetModel")
INPUTS=("devemo_combined")
CLASSES=("binary" "all")

for cls in "${CLASSES[@]}"; do
    for input in "${INPUTS[@]}"; do
        for model in "${MODELS[@]}"; do
            echo ""
            echo "--- Model=$model Input=$input Class=$cls Epochs=$EPOCHS Loops=$LOOPS NumFrames=$RUN_NUM_FRAMES ---"
            python main.py --model "$model" --input "$input" --mode both --epochs "$EPOCHS" \
                --train-frame-selection uniform --test-frame-selection uniform \
                --num-frames "$RUN_NUM_FRAMES" --class-split "$cls" --loop "$LOOPS"
        done
    done
done

echo "--- Average Model Performance ---"
python misc/average_runs.py -n "$LOOPS"

echo "======================================================"
echo "All test runs completed successfully!"
echo "======================================================"
