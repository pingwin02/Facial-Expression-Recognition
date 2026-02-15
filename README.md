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

### Option 1: Run directly with Python (`main.py`)

Train:

```bash
python main.py --input devemo+ --mode train --epochs 100 --model TransferModel
```

Evaluate:

```bash
python main.py --input devemo+ --mode eval --model TransferModel
```

Useful values:

- `--input`: `devemo`, `devemo+`, `fer2013` and other
- `--model`: `TransferModel`, `BinaryModel` and other

### Option 2: Run with bash script (`train_eval.sh`)

The script runs both phases (`train` then `eval`) and supports menu index selection.

Example (TransferModel + devemo+):

```bash
./train_eval.sh -m 1 -i 0
```

Parameters:

- `-e` epochs (default `100`)
- `-m` model index from menu
- `-i` input index from menu

## Useful commands

- **Run script in background**: To run a script in the background, you can use the `nohup` command:
  ```bash
  nohup ./train_eval.sh &> out.log &
  ```
- **Check GPU & RAM usage**: To check the GPU & RAM usage, you can use the `nvidia-smi` and `free` command:
  ```bash
  watch -n 0.5 -d "nvidia-smi && free -h"
  ```
- **Check logs**: To check the logs of a running script, you can use the `tail` command:
  ```bash
  tail -f out.log
  ```
- **Kill a process**: To kill a process, you can use the `kill` command:
  ```bash
  kill -9 <pid>
  ```
  You can find the PID of a process using the `ps` command:
  ```bash
  ps -ef | grep python
  ```
  or if you want to kill all processes with a specific name:
  ```bash
  pkill -f <process_name>
  ```