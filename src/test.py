import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft

def calculate_time_features(csi_segment):
    max_value = np.max(csi_segment)
    min_value = np.min(csi_segment)
    mean_value = np.mean(csi_segment)
    median_value = np.median(csi_segment)
    std_dev = np.std(csi_segment)
    skewness = skew(csi_segment)
    kurt = kurtosis(csi_segment)
    return [max_value, min_value, mean_value, median_value, std_dev, skewness, kurt]

def calculate_frequency_features(csi_data, fs):
    freqs = fft(csi_data)
    low_energy = np.sum(np.abs(freqs[:int(0.7 * fs)]))
    activity_energy = np.sum(np.abs(freqs[int(0.3 * fs):int(2.0 * fs)]))
    high_energy = np.sum(np.abs(freqs[int(0.7 * fs):int(10.0 * fs)]))
    return [low_energy, activity_energy, high_energy]
