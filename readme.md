# Diffusion Models for Replay in Continual Learning

## Introduction

***This repository is under construction***

## Requirements

- Python 3.9.16
- PyTorch 2.0.1
- Diffusers 0.16.1
- avalanche-lib 0.3.1
- torch-fidelity 0.3.0

## Usage

### Generative Replay With VAE

``` shell
python generative_replay_vae.py [-h] [--image_size IMAGE_SIZE] [--batch_size BATCH_SIZE] [--channels CHANNELS] [--epochs_generator EPOCHS_GENERATOR] [--epochs_solver EPOCHS_SOLVER] [--generator_lr GENERATOR_LR] [--generator_weight_decay GENERATOR_WEIGHT_DECAY] [--solver_lr SOLVER_LR] [--increasing_replay_size INCREASING_REPLAY_SIZE] [--replay_size REPLAY_SIZE] [--seed SEED] [--cuda CUDA] [--debug]
```

### Generative Replay With Diffusion Models

``` shell
python generative_replay_diffusion.py [-h] [--image_size IMAGE_SIZE] [--batch_size BATCH_SIZE] [--channels CHANNELS] [--epochs_generator EPOCHS_GENERATOR] [--epochs_solver EPOCHS_SOLVER] [--generator_lr GENERATOR_LR] [--generator_weight_decay GENERATOR_WEIGHT_DECAY] [--solver_lr SOLVER_LR] [--increasing_replay_size INCREASING_REPLAY_SIZE] [--replay_size REPLAY_SIZE] [--seed SEED] [--cuda CUDA] [--debug]
```

