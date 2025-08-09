# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research implementation of "Rethinking Federated Learning with Domain Shift: A Prototype View" (CVPR 2023). The codebase implements Federated Prototypes Learning (FPL) and other federated learning algorithms for scenarios with domain shift.

## Running Experiments

### Basic Execution
```bash
python main.py --model fpl --dataset fl_digits --communication_epoch 100 --local_epoch 10 --parti_num 10
```

### Available Models
- `fpl` - Federated Prototypes Learning (main contribution)
- `fedavg` - FedAvg baseline
- `fedprox` - FedProx algorithm
- `moon` - MOON algorithm
- `local` - Local training only

### Available Datasets
- `fl_digits` - Digits dataset with domain shift (MNIST, SVHN, USPS)
- `fl_officecaltech` - Office-Caltech dataset with domain shift

### Key Parameters
- `--communication_epoch` - Number of federated rounds
- `--local_epoch` - Local epochs per client per round
- `--parti_num` - Number of participating clients
- `--device_id` - GPU device ID
- `--seed` - Random seed for reproducibility

## Code Architecture

### Core Components
- `main.py` - Entry point, handles argument parsing and experiment setup
- `models/` - Federated learning algorithms
  - `fpl.py` - Main FPL implementation with prototype clustering
  - `fedavg.py` - FedAvg baseline
  - `utils/federated_model.py` - Base class for federated models
- `datasets/` - Dataset handling for federated scenarios
  - `digits.py` - Digits dataset with domain partitioning
  - `officecaltech.py` - Office-Caltech dataset handling
  - `utils/federated_dataset.py` - Base federated dataset utilities
- `backbone/` - Neural network architectures (ResNet, CNN, etc.)
- `utils/` - Training utilities, logging, and configuration

### Key Design Patterns
- Federated models inherit from `FederatedModel` base class
- Each model implements `loc_update()` for local training and `ini()` for initialization
- Datasets are partitioned across clients to simulate federated scenarios
- Prototype aggregation uses FINCH clustering algorithm

### Dependencies
Install requirements:
```bash
pip install -r requirements.txt
```

Main dependencies: torch, torchvision, numpy, scikit-learn, scipy, tqdm

## Best Arguments Configuration

The system uses predefined best hyperparameters stored in `utils/best_args.py` that override command-line arguments for reproducibility. These are automatically applied based on the dataset-model combination.

## Results and Checkpoints

- Results are saved in `data/` directory organized by dataset and method
- Checkpoints stored in `checkpoint/` directory
- Experiment logs include accuracy metrics and hyperparameters