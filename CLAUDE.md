# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuroLM is a universal multi-task foundation model that bridges language and EEG signals by treating EEG as a "foreign language" for LLMs. It's built on GPT-2 architecture with a vector-quantized neural tokenizer that converts EEG signals into discrete tokens.

## Key Architecture

### Core Components
- **VQ Module** (`model/model_vq.py`): Frozen neural tokenizer that encodes EEG → discrete tokens
- **NeuroLM** (`model/model_neurolm.py`): Main model supporting Base/Large/XL variants
- **Neural Transformer** (`model/model_neural_transformer.py`): Multi-channel autoregressive processing
- **Metrics** (`metrics/`): Comprehensive evaluation framework

### Model Variants
- NeuroLM-B (Base): 110M parameters
- NeuroLM-L (Large): 344M parameters  
- NeuroLM-XL (Extra Large): 1.7B parameters

## Development Commands

### Environment Setup
```bash
# Using pixi (preferred)
pixi install

# Using conda
conda create -n NeuroLM python=3.12
conda activate NeuroLM
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install transformers datasets==2.21.0 tiktoken wandb h5py einops pandas scikit-learn
```

### Training Pipeline
```bash
# 1. Prepare datasets
python dataset_maker/prepare_TUH_pretrain.py
python text_dataset_maker/prepare.py

# 2. Train VQ tokenizer
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=3 train_vq.py \
    --dataset_dir /path/to/dataset --out_dir /path/to/save --wandb_log

# 3. Pre-train NeuroLM
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=3 train_pretrain.py \
    --dataset_dir /path/to/dataset --out_dir /path/to/save \
    --tokenizer_path checkpoints/VQ.pt

# 4. Instruction tuning
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=3 train_instruction.py \
    --dataset_dir /path/to/dataset --out_dir /path/to/save \
    --NeuroLM_path checkpoints/NeuroLM-B.pt
```

## Critical Implementation Details

### EEG Processing
- Standard 10-20 channel system (19 channels)
- Preprocessing: 0.1-75 Hz filtering, 50/60 Hz notch, 200 Hz resampling
- Data stored as pickle files with μV measurements

### Training Configuration
- Hardware: 3x RTX 6000 Ada GPUs
- Distributed training via PyTorch DDP
- Uses Weights & Biases for experiment tracking
- OMP_NUM_THREADS=1 required for multi-GPU training

### Key Training Scripts
- `train_vq.py`: Trains the neural tokenizer
- `train_pretrain.py`: Multi-channel autoregressive pre-training
- `train_instruction.py`: Multi-task instruction tuning
- `sample.py`: Inference/generation script

### Dataset Structure
- Pre-training: TUH dataset (~25,000 hours)
- Downstream: TUAB, TUEV, TUSL, HMC, Workload, SEED
- Custom dataset makers in `dataset_maker/` and `text_dataset_maker/`

## Important Notes
- No automated tests or linting setup - this is research code
- Model checkpoints already downloaded in `checkpoints/` directory
- Uses `pixi` package manager (see `pixi.toml`)