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