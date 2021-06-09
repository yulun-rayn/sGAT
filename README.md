# Spatial Graph Attention

This repository is the official implementation of Spatial Graph Attention Network (sGAT) within the paper [**Spatial Graph Attention and Curiosity-driven Policy for Antiviral Drug Discovery**](https://arxiv.org/abs/2106.02190). The Reinforcement Learning architecture built upon sGAT can be found in the main repository [here](https://github.com/njchoma/DGAPN).


## Installation

### 1. Create conda environment
```bash
conda create -n sgat-env --file requirements.txt
conda activate sgat-env
```

### 2. Install learning library
- Pytorch **1.8**.0 (https://pytorch.org/get-started/locally/)
- Pytorch Geometric **1.6**.3 (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

  \* *make sure to install the right versions for your toolkit*


## Run
Once the environment is set up, the function call to train & evaluate sGAT is:

```bash
./main.sh
```

A list of flags may be found in `main.sh` and `src/main.py` for experimentation with different network parameters.
