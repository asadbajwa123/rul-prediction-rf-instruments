# Configuration settings for the rul-prediction-rf-instruments project

# Paths
DATA_PATH = './data/'
MODEL_PATH = './models/'
RESULTS_PATH = './results/'

# Dataset settings
DATASET_NAME = 'dataset.csv'
SPLIT_RATIO = 0.8

# Model hyperparameters
HYPERPARAMETERS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}