# Facial Expression Recognition

The goal of this project is to analyze existing algorithms for emotion
recognition from video recordings based on facial expressions and to implement
selected methods. Additionally, algorithms proposed by the supervisor are
implemented and their accuracy is compared with other commonly used approaches.

## Installation

To install the required dependencies, run the following command:

```bash
conda create -n fer python=3.10 -y && \
conda activate fer && \
pip install -r requirements.txt && \
conda install -c conda-forge libstdcxx-ng gcc_linux-64 gxx_linux-64 -y
```

## Running the project

Before running, activate environment and enter project directory:

```bash
conda activate fer
cd Facial-Expression-Recognition
```

### Usage

All training/evaluation logic is handled by `main.py`. Run without arguments for interactive mode (prompts for all
parameters with defaults), or pass arguments directly.

For full help and available options:

```bash
python main.py --help
```

Arguments accept both **names** and **numeric indices** (e.g. `--model 0` or `--model ResNetModel`).

### Examples

Train and evaluate ResNetModel on devemo_combined, 5 frames, binary classes:

```bash
python main.py --model ResNetModel --input devemo_combined --mode both --epochs 100 \
    --train-frame-selection uniform --test-frame-selection uniform --num-frames 5 --class-split binary
```

Same using numeric indices:

```bash
python main.py --model 0 --input 4 --mode 2 --epochs 100 \
    --train-frame-selection 0 --test-frame-selection 0 --num-frames 5 --class-split 0
```

Interactive mode (prompts for everything):

```bash
python main.py
```

### Parameters

| Flag                      | Description                      | Default       |
|---------------------------|----------------------------------|---------------|
| `--model`                 | Model name or index              | (interactive) |
| `--input`                 | Dataset name or index            | (interactive) |
| `--mode`                  | `train`, `eval`, `both` (or 0-2) | `both`        |
| `--epochs`                | Number of training epochs        | 100           |
| `--train-frame-selection` | Frame selection for training     | `uniform`     |
| `--test-frame-selection`  | Frame selection for testing      | same as train |
| `--num-frames`            | Frames per video                 | 5             |
| `--class-split`           | `binary` or `all` (or 0-1)       | `binary`      |
| `--loop`                  | Number of full run loops         | 1             |
| `--no-cache`              | Disable dataset caching          | off           |

### Quick regression check

Run the bundled smoke-test script:

```bash
./run_tests.sh
```

Useful variants:

```bash
# run a short debug pass (1 loop, 10 epochs)
./run_tests.sh --debug

# run in background and save PID/logs in the project root
./run_tests.sh --detached

# combine both
./run_tests.sh --detached --debug
```

### Optional: Weights & Biases (wandb) logging

Training supports optional wandb logging. Put your API key in `.env` in the project root:

```bash
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=facial-expression-recognition
# optional
# WANDB_ENTITY=your_team_or_user
# WANDB_MODE=online
# WANDB_RUN_NAME=my_custom_run_name
```

If `WANDB_API_KEY` is missing, training runs normally without wandb.

## Misc scripts

Helper scripts in `misc/` for analysis and visualization:

| Script                           | Description                                                                             | Example                                                   |
|----------------------------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `average_runs.py`                | Averages metrics across multiple training runs in `output/` and generates summary plots | `python misc/average_runs.py`                             |
| `dataset_statistics.py`          | Generates class distribution plots for devemo/devemo+ datasets                          | `python misc/dataset_statistics.py`                       |
| `temporal_overlay_preview.py`    | Renders temporal encoding preview (sampled frames + diff layers) for a video            | `python misc/temporal_overlay_preview.py --video PATH`    |
| `veatic_single_film_timeline.py` | Plots arousal/valence frame-label timeline for a single VEATIC film                     | `python misc/veatic_single_film_timeline.py --video-id 0` |
| `view_cache_pkl_frames.py`       | Visualizes frames stored in cache `.pkl` files                                          | `python misc/view_cache_pkl_frames.py`                    |

## Useful commands

- **Run script in background and save PID**:
  ```bash
  nohup ./run_tests.sh &> out.log & echo $! > out.pid
  ```
- **Use built-in detached mode**:
  ```bash
  ./run_tests.sh --detached
  ```
- **Kill script and all its Python subprocesses**:
  ```bash
  kill -9 -$(cat out.pid)
  ```
- **Check if process or its subprocesses are running**:
  ```bash
  pgrep -g $(cat out.pid) -a
  ps -ef | grep $(cat out.pid)
  ```
- **Check GPU & RAM usage**:
  ```bash
  watch -n 0.5 -d "nvidia-smi && free -h"
  ```
- **Check logs**:
  ```bash
  tail -F out.log
  ```