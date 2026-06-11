#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE}")" && pwd)"
SELF_SCRIPT="${SCRIPT_DIR}/run_tests.sh"
RUN_TESTS_CONFIG="${SCRIPT_DIR}/run_tests.config.sh"

DETACHED=0
CACHE_ONLY=0
LOG_FILE="${SCRIPT_DIR}/out.log"
PID_FILE="${SCRIPT_DIR}/out.pid"

cleanup_pid_file() {
    local current_pid=""
    if [[ -f "${PID_FILE}" ]]; then
        current_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    fi

    if [[ "${current_pid}" == "$$" ]]; then
        rm -f "${PID_FILE}"
    fi
}

get_frames() {
    if [[ "$1" == "veatic" ]]; then
        echo 300
    else
        echo 5
    fi
}

uses_configured_classes() {
    case "$1" in
        devemo|devemo+|devemo_combined)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

requires_configured_classes() {
    for input in "${INPUTS[@]}"; do
        if uses_configured_classes "${input}"; then
            return 0
        fi
    done
    return 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --detached)   DETACHED=1; shift ;;
        --cache-only) CACHE_ONLY=1; shift ;;
        --help|-h)
            echo "Usage: ./run_tests.sh [--detached] [--cache-only]"
            echo "  --detached    Run the test suite in the background"
            echo "  --cache-only  Run main.py with --cache-only"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ${DETACHED} -eq 1 ]]; then
    DETACHED_CMD=("${SELF_SCRIPT}")
    [[ ${CACHE_ONLY} -eq 1 ]] && DETACHED_CMD+=(--cache-only)

    nohup setsid "${DETACHED_CMD[@]}" &> "${LOG_FILE}" &
    echo $! > "${PID_FILE}"
    echo "Started run_tests.sh in background. PID: ${PID_FILE}"
    exit 0
fi

cd "${SCRIPT_DIR}"

if [[ -f "${PID_FILE}" ]] && [[ "$(cat "${PID_FILE}")" == "$$" ]]; then
    trap cleanup_pid_file EXIT
fi

if [[ ${CACHE_ONLY} -eq 1 ]]; then
    if [[ -z "${DATASET_CACHE_TO_BUILD:-}" ]]; then
        echo "DATASET_CACHE_TO_BUILD must be set when using --cache-only." >&2
        exit 1
    fi

    RUN_FRAMES=$(get_frames "${DATASET_CACHE_TO_BUILD}")

    echo "Building cache for ${DATASET_CACHE_TO_BUILD} with ${RUN_FRAMES} frames..."
    python main.py --model 0 --input "${DATASET_CACHE_TO_BUILD}" --mode both \
        --epochs 1 --train-frame-selection uniform --test-frame-selection uniform \
        --num-frames "${RUN_FRAMES}" --class-split "binary" --loop 1 --cache-only
    exit 0
fi

echo "======================================================"
echo "Starting all test runs..."
echo "======================================================"

if [[ ! -f "${RUN_TESTS_CONFIG}" ]]; then
    echo "Missing ${RUN_TESTS_CONFIG}." >&2
    echo "See README.md for the expected run_tests.config.sh format." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${RUN_TESTS_CONFIG}"

if [[ -z "${EPOCHS:-}" ]] || [[ -z "${LOOPS:-}" ]] || [[ ${#MODELS[@]} -eq 0 ]] || [[ ${#INPUTS[@]} -eq 0 ]]; then
    echo "${RUN_TESTS_CONFIG} must define EPOCHS, LOOPS, and non-empty MODELS and INPUTS arrays." >&2
    echo "See README.md for the expected run_tests.config.sh format." >&2
    exit 1
fi

if requires_configured_classes && [[ ${#CLASSES[@]} -eq 0 ]]; then
    echo "${RUN_TESTS_CONFIG} must define a non-empty CLASSES array when INPUTS includes devemo/devemo+/devemo_combined." >&2
    echo "See README.md for the expected run_tests.config.sh format." >&2
    exit 1
fi

for input in "${INPUTS[@]}"; do
    if uses_configured_classes "${input}"; then
        input_classes=("${CLASSES[@]}")
    else
        input_classes=("binary")
    fi

    for cls in "${input_classes[@]}"; do
        RUN_FRAMES=$(get_frames "${input}")

        for model in "${MODELS[@]}"; do
            echo ""
            echo "--- Model=$model Input=$input Class=$cls Epochs=$EPOCHS Loops=$LOOPS NumFrames=$RUN_FRAMES ---"
            python main.py --model "$model" --input "$input" --mode both --epochs "$EPOCHS" \
                --train-frame-selection uniform --test-frame-selection uniform \
                --num-frames "$RUN_FRAMES" --class-split "$cls" --loop "$LOOPS"
        done
    done
done

echo "--- Average Model Performance ---"
python misc/average_runs.py -n "$LOOPS"
