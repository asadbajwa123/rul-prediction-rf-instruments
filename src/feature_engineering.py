import pandas as pd
import numpy as np


def statistical_features(data):
    # Calculate statistical features
    mean = np.mean(data)
    std = np.std(data)
    median = np.median(data)
    min_val = np.min(data)
    max_val = np.max(data)
    return {'mean': mean, 'std': std, 'median': median, 'min': min_val, 'max': max_val}


def trend_features(data):
    # Calculate trend features
    trend = np.polyfit(range(len(data)), data, 1)[0]
    return {'trend_slope': trend}


def degradation_indicators(data):
    # Example degradation indicator calculations
    degradation_rate = np.diff(data) / data[:-1]
    return {'degradation_rate': degradation_rate}


def feature_selection(data, target, threshold=0.1):
    # Select features based on correlation with the target
    correlations = data.corrwith(target)
    selected_features = correlations[correlations.abs() > threshold].index.tolist()
    return selected_features
