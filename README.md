# Keras MLP Playground

This project is designed for experimenting with hyperparameters in Multi-Layer Perceptron (MLP) neural networks. It provides a structured way to train models with different configurations and automatically tracks all experiments, making it easy to compare results and analyze the impact of various hyperparameter choices.

## Features

- **Experiment Tracking**: Each training run is automatically saved with timestamps and performance metrics
- **Comprehensive Logging**: Detailed summaries of model architecture, hyperparameters, and results
- **Visualization**: Automatic generation of training/validation curves for each experiment
- **Dataset Management**: Support for multiple datasets with easy selection
- **Model Checkpointing**: Saves best performing models during training
- **Standardized Preprocessing**: Automatic feature scaling and data normalization

## Project Structure

```
├── main.py              # Main training script
├── utils.py            # Helper functions
├── dataset/           # Directory for training datasets
└── runs/              # Experiment outputs
    └── [timestamp]-train-[acc]-test-[acc]/
        ├── code/      # Snapshot of the code used
        ├── model/     # Saved model files
        ├── plot/      # Training history visualizations
        └── summary/   # Detailed experiment summary
```

## Usage

1. Place your dataset(s) in the `dataset/` directory (CSV format)
2. Run the training script:
   ```bash
   python main.py
   ```
3. If multiple datasets are available, select one when prompted
4. Results will be automatically saved in the `runs/` directory

## Experiment Results

Each experiment run creates a new directory with the following naming convention:
`[date]-[time]-train-[training_accuracy]-test-[test_accuracy]`

The directory contains:
- Complete model code snapshot
- Training history plots
- Best model weights
- Detailed summary of configuration and results

## Requirements

- TensorFlow/Keras
- NumPy
- Matplotlib
- Scikit-learn
