import numpy as np
import scipy

from vitalwave import basic_algos

def clean_accelerometer_signal(arr:np.ndarray, dwt_transform:str = 'bior4.4', dlevels=9,
                               cutoff_low:int = 1, cutoff_high:int = 9) -> np.ndarray:
    """
    Clean and preprocess an accelerometer signal using wavelet transformation and min-max normalization.

    Parameters:
    arr: Input accelerometer signal.
    dwt_transform: Wavelet transform type (default: 'bior4.4').
    dlevels: Number of decomposition levels in the wavelet transform (default: 9).
    cutoff_low: Low-frequency cutoff for wavelet transform (default: 1).
    cutoff_high: High-frequency cutoff for wavelet transform (default: 9).

    Returns:
    numpy.ndarray: Cleaned and normalized accelerometer signal.
    """
    acc_cleaned = basic_algos.wavelet_transform_signal(arr = arr, dwt_transform = dwt_transform, dlevels=dlevels,
                                                       cutoff_low=cutoff_low, cutoff_high=cutoff_high)

    acc = basic_algos.min_max_normalize(arr=acc_cleaned)
    return acc

def calculate_gravity_and_movement_xyz(arr_ac_x:np.ndarray, arr_ac_y:np.ndarray, arr_ac_z:np.ndarray, fs:float):
    """
    Calculate gravity and movement components of accelerometer data.

    Parameters:
    arr_ac_x: Raw accelerometer data along the X-axis.
    arr_ac_y: Raw accelerometer data along the Y-axis.
    arr_ac_z: Raw accelerometer data along the Z-axis.
    fs (float): Sampling frequency.

    Returns:
    numpy.ndarray: Gravity component of the data for each axis (X, Y, Z).
    """
    gravity_x_filtered = basic_algos.butter_filter(arr_ac_x, n=4,  fs=fs, wn=1, filter_type='low')
    gravity_y_filtered = basic_algos.butter_filter(arr_ac_y, n=4,  fs=fs, wn=1, filter_type='low')
    gravity_z_filtered = basic_algos.butter_filter(arr_ac_z, n=4,  fs=fs, wn=1, filter_type='low')

    gravity = np.column_stack((gravity_x_filtered, gravity_y_filtered, gravity_z_filtered))

    return gravity

def calculate_gravity_statistics(gravity_data:np.ndarray):
    """
    Calculate statistics and additional features for gravity data.

    Parameters:
    gravity_data: Input gravity data.

    Returns:
    _GravityStatisticsCalculator: An instance of the _GravityStatisticsCalculator class
    containing calculated statistics and additional features for the input gravity data.
    """
    calculator = _GravityStatisticsCalculator(gravity_data)
    return calculator

class _GravityStatisticsCalculator:
    def __init__(self, gravity):
        self.gravity = gravity
        self.mean = np.mean(self.gravity, axis=1)
        self.median = np.median(self.gravity, axis=1)
        self.std_dev = np.std(self.gravity, axis=1)
        std_dev_ddof1 = np.std(self.gravity, axis=1, ddof=1)
        self.coeff_var = std_dev_ddof1 / self.mean
        self.percentile_25 = np.percentile(self.gravity, 25, axis=1)
        self.percentile_75 = np.percentile(self.gravity, 75, axis=1)
        self.min_val = np.min(self.gravity, axis=1)
        self.max_val = np.max(self.gravity, axis=1)

        # Calculate additional features
        filt_g = scipy.signal.savgol_filter(np.copy(self.gravity), 5, 2, deriv=1)
        self.num_sign_changes_filt = (np.diff(np.sign(filt_g), axis=1) != 0).sum(axis=1)
        self.num_sign_changes_orig = (np.diff(np.sign(self.gravity), axis=1) != 0).sum(axis=1)

    def __str__(self):
        return f"Mean: {self.mean}\n" \
               f"Median: {self.median}\n" \
               f"Standard Deviation: {self.std_dev}\n" \
               f"Coefficient of Variation: {self.coeff_var}\n" \
               f"25th Percentile: {self.percentile_25}\n" \
               f"75th Percentile: {self.percentile_75}\n" \
               f"Min Value: {self.min_val}\n" \
               f"Max Value: {self.max_val}\n" \
               f"Num Sign Changes (Filtered): {self.num_sign_changes_filt}\n" \
               f"Num Sign Changes (Original): {self.num_sign_changes_orig}"


def extract_frequency_domain_features(movement):
    """
    Extract frequency domain features from input movement data.

    Parameters:
    data (numpy.ndarray): Input data.

    Returns:
    list of frequency domain features encapsulated within a class.
    """
    features = _FrequencyDomainFeatureExtractor(movement=movement)
    return features

class _FrequencyDomainFeatureExtractor:
    def __init__(self, movement):
        self.movement = movement

        # Calculate statistical features
        self.skewness = scipy.stats.skew(self.movement, axis=1)
        self.kurtosis = scipy.stats.kurtosis(self.movement, axis=1)
        self.sum_of_squares = np.sum(np.square(self.movement), axis=1)

        # Calculate frequency-domain features
        abs_fft_data = np.abs(np.fft.rfft(self.movement, axis=1))
        self.mean_magnitudes = np.mean(abs_fft_data, axis=1)
        self.std_magnitudes = np.std(abs_fft_data, axis=1)
        freq_values = np.fft.rfftfreq(self.movement.shape[1])
        self.max_magnitudes = np.abs(np.fft.rfft(self.movement, axis=1)).max(axis=1)
        self.dominant_frequencies = freq_values[np.abs(np.fft.rfft(self.movement, axis=1)).argmax(axis=1)]
        sums = np.sum(np.abs(np.fft.rfft(self.movement, axis=1)), axis=1)
        sums = np.where(sums, sums, 1.)
        self.spectral_centroid = np.sum(freq_values * abs_fft_data, axis=1) / sums
        self.total_power = np.sum(abs_fft_data ** 2, axis=1)

        # zero-crossings
        filt_m = scipy.signal.savgol_filter(np.copy(self.movement), 5, 2, deriv=1)
        self.num_sign_changes_filt = (np.diff(np.sign(filt_m), axis=1) != 0).sum(axis=1)
        self.num_sign_changes_orig = (np.diff(np.sign(self.movement), axis=1) != 0).sum(axis=1)

    def __str__(self):
        return f"Skewness: {self.skewness}\n" \
               f"Kurtosis: {self.kurtosis}\n" \
               f"Sum of Squares: {self.sum_of_squares}\n" \
               f"Mean Magnitudes: {self.mean_magnitudes}\n" \
               f"Std Magnitudes: {self.std_magnitudes}\n" \
               f"Dominant Frequencies: {self.dominant_frequencies}\n" \
               f"Max Magnitudes: {self.max_magnitudes}\n" \
               f"Spectral Centroid: {self.spectral_centroid}\n" \
               f"Total Power: {self.total_power}\n"\
               f"Num Sign Changes (Filtered): {self.num_sign_changes_filt}\n" \
               f"Num Sign Changes (Original): {self.num_sign_changes_orig}"

def calculate_polynomial_fit(data:np.ndarray):
    """
    Calculate polynomial fit features based on input data.

    Parameters:
    - data (numpy.ndarray): Input data matrix.

    Returns:
    tuple: Polynomial fit features including pitch, roll, and theta.
    """
    pitch = np.arctan((data[0, :] / (np.square(data[1, :]) + np.square(data[2, :]))))
    roll = np.arctan((data[1, :] / (np.square(data[0, :]) + np.square(data[2, :]))))
    theta = np.arctan(((np.square(data[1, :]) + np.square(data[0, :])) / data[2, :]))

    p1 = np.ravel(np.polyfit(x = np.arange(len(pitch)), y = pitch, deg = 1))
    p2 = np.ravel(np.polyfit(x = np.arange(len(pitch)), y = pitch, deg = 2))
    r1 = np.ravel(np.polyfit(np.arange(len(roll)), roll, 1))
    r2 = np.ravel(np.polyfit(np.arange(len(roll)), roll, 2))
    t1 = np.ravel(np.polyfit(np.arange(len(theta)), theta, 1))
    t2 = np.ravel(np.polyfit(np.arange(len(theta)), theta, 2))

    return p1, p2, r1, r2, t1, t2

def axes_corr(data:np.ndarray, size:int = 2):
    """
    Compute correlation between all axes (and the magnitudes) for a given data matrix.

    Parameters:
    - data: Input data matrix.
    - size: The number of axes to consider. Default is 2.

    Returns:
    Correlation coefficients.
    """
    corr = []
    for i in range(size):
        for j in range(size):
            if j <= i:
                continue
            x = data[:, i]
            y = data[:, j]
            corr.append((np.mean((x * y)) - np.mean(x) * np.mean(y)) / (np.std(x) * np.std(y)))

    x = data[:, -2]
    y = data[:, -1]
    corr.append((np.mean((x * y)) - np.mean(x) * np.mean(y)) / (np.std(x) * np.std(y)))

    return np.array(corr)

def get_activity_features(arr_ac_x:np.ndarray, arr_ac_y:np.ndarray, arr_ac_z:np.ndarray, fs:float, size:int=6):
    """
    Calculate activity features for a single sensor with three degrees of movement.

    Parameters:
    - arr_ac_x: Accelerometer data for the X-axis.
    - arr_ac_y: Accelerometer data for the Y-axis.
    - arr_ac_z: Accelerometer data for the Z-axis.
    - fs: Sampling frequency.
    - size (int, optional): Size parameter for correlations calculation.

    Returns:
    - gravity_statistics
    - freq_domain_features
    - poly_fit_features
    - correlations
    """
    # start of the original code
    gravity = calculate_gravity_and_movement_xyz(arr_ac_x=arr_ac_x,
                                                 arr_ac_y=arr_ac_y,
                                                 arr_ac_z=arr_ac_z,
                                                 fs=fs)

    movement = np.column_stack((arr_ac_x, arr_ac_y, arr_ac_z))

    # gravity features;
    gravity_statistics = calculate_gravity_statistics(gravity.T)

    # movement features;
    freq_domain_features = extract_frequency_domain_features(movement.T)

    # polynomial fit features
    poly_fit_features = calculate_polynomial_fit(movement)

    # correlations between the axes
    correlations = axes_corr(movement.T, size=size)

    return gravity_statistics, freq_domain_features, poly_fit_features, correlations


