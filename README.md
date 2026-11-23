# fl-health

# Federated Learning for Healthcare: Diabetes Readmission Prediction

This project implements a privacy-preserving federated learning framework for predicting diabetes patient readmission using the Diabetes 130-US Hospitals dataset. The system enables collaborative model training across multiple simulated hospital data silos without requiring direct data sharing.

## Key Features

- **Privacy-Preserving ML**: Implements federated learning with differential privacy (DP-SGD) and weight update clipping
- **Healthcare Focus**: Designed specifically for sensitive medical data with HIPAA/GDPR considerations
- **Multi-Framework Support**: Compatible with Flower, TensorFlow Federated, PySyft, OpenFL, and Substra
- **Comprehensive Monitoring**: Tracks both node-level and aggregated model performance metrics
- **Realistic Simulation**: 5 simulated hospital silos with non-IID data distribution

## Dataset

- **Diabetes 130-US Hospitals Dataset** (100,000+ admissions across 130 hospitals)
- Features: demographics, diagnoses, medications, readmission status
- Source: [Kaggle](https://www.kaggle.com/datasets/brandao/diabetes)

## Installation

1. Clone this repository
2. Create conda environment: `conda env create -f environment.yml`
3. Activate environment: `conda activate fl-health`

## Project Structure

fl-health/
├── envs/ # Virtual environments
├── datasets/diabetes/ # Data storage
| └── processed_silos/ # Silo data partitions
├── notebooks/ # Jupyter analysis
├── scripts/ # Preprocessing utilities
| ├── client_logs/ # Node-level monitoring
| └── server_logs/ # Aggregated metrics
└── fl_frameworks/ # Framework implementations

## Usage

1. Preprocess data: `python scripts/preprocess_silo.py`
2. Run federated training: `python fl_frameworks/flower_example/server.py`
3. Launch clients: `python fl_frameworks/flower_example/client.py silo_1.csv`

## Results

- Baseline centralized model: 87.1% accuracy
- Federated model: 88.77% accuracy across silos
- Privacy guarantees: DP-SGD (noise_multiplier=1.0) with weight clipping

## Contributors

- **Irshad Ahmad Oruzgani** (Primary Developer)
- **Imanbek Bagan Talgatkyzy** (Head of Practice)

Institution: Al-Farabi Kazakh National University, Faculty of Information Technology
