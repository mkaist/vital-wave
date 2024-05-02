

import numpy as np
from scipy.stats import pearsonr
from scipy.signal import welch, resample, kurtosis, skew, linregress 
from vitalwave.basic_algos import extract_waveforms
import math 

#ADD TYPESETTING
def Absolute_Signal_to_noise_Ratio(arr):
    """
    Calculate the Signal-to-Noise Ratio (SNR) of a filtered signal.
    SNR is defined as the ratio of the standard deviation of the absolute signal to
    the standard deviation of the signal itself.
    
    Parameters:
    - arr: numpy array of the filtered signal.
    
    Returns:
    - SNR: The calculated Signal-to-Noise Ratio.
    """
    # Calculate the standard deviation of the signal and its absolute value
    
    return np.std(arr)/np.std(np.abs(arr))



def Cardiac_Pulse_Power(arr, peaks, fs, each_slice_length=100):
    """
    Calculate the average energy of heart cycles extracted from an array based on specified peaks.

    Parameters:
    - arr (numpy array): The input signal array.
    - peaks (list): Indices of detected peaks in the signal.
    - fs (int): Sampling frequency of the signal, used to define the default slice length.
    - each_slice_length (int): Number of samples to include in each heart cycle slice. Default is 100 samples.

    Returns:
    - float: The average energy of the extracted heart cycles.
    """
    
    # Calculate interval length based on sampling frequency and slice length
    interval_length = each_slice_length * fs   

    #USE CODE FROM BASIC ALGOS
    # Extract cardiac cycles using numpy slicing for efficiency
    cardiac_cycles = [arr[max(0, peak - interval_length // 2):min(len(arr), peak + interval_length // 2)] for peak in peaks]
    
    # Calculate the energy for each cardiac cycle using numpy operations
    energies = [np.sum(np.square(cycle)) for cycle in cardiac cycles]
    
    # Calculate the mean energy of the cardiac cycles
    average_energy = np.mean(energies) if energies else 0  # Return 0 if no cycles are extracted
    return average_energy



def Lempel_Ziv_Complexity(arr):
    """
    Compute the Lempel-Ziv complexity of a signal.
    
    Parameters:
    - arr: The input signal.
    
    Returns:
    - LZ complexity value.
    """
    # Convert signal to a binary sequence based on its median
    median_value = np.median(arr)
    sequence = [1 if x > median_value else 0 for x in arr]
    
    # Convert sequence to string
    s = ''.join(map(str, sequence))
    
    # Initialize the complexity and the subsequence
    complexity = 1
    subsequence = s[0]
    
    for i in range(1, len(s)):
        if s[i] not in subsequence:
            complexity += 1
            subsequence += s[i]
        else:
            subsequence = s[i]
    
    # Normalize the complexity to the interval [0,1]
    normalized_complexity = complexity / len(s)  
    return normalized_complexity




def Kurtosis_Index(arr):
    """
    Calculate the kurtosis of the input array. Kurtosis is a measure of the "tailedness" of the probability distribution
    of a real-valued random variable. 
    
    Parameters:
    - arr: The input array for which to calculate the kurtosis.
    
    Returns:
    - float: The kurtosis index of the array.
    """
    kurtosis_index = kurtosis(arr)
    return kurtosis_index



def Signal_Consistency_Range(arr):
    """
    Calculate the interquartile range (IQR) of an array. The IQR is the difference between
    the 75th and 25th percentiles of the array, providing a measure of statistical dispersion.
    
    Parameters:
    - arr: The input array for which to calculate the IQR.
    
    Returns:
    - float: The interquartile range of the array.
    """
    # Calculate the 25th and 75th percentiles and compute the IQR
    iqr_value = np.percentile(arr, 75) - np.percentile(arr, 25)
    return iqr_value


def Clipping_Quality_Index(arr, locs, fs, threshold=1):
    """
    Calculate the Clipping Quality Index (SQI) for waveforms extracted from a signal.

    Parameters:
    - arr (numpy array): The array containing the signal data.
    - locs (list of integers): The indices of peaks around which waveforms are extracted.
    - fs (int): The sampling frequency of the signal.
    - threshold (int or float): The value above which the difference between consecutive 
      samples is considered to indicate clipping.

    Returns:
    - float: The average proportion of non-clipped segments across all analyzed waveforms.
    """

    waveforms = extract_waveforms(arr, locs, fs) #USE VITALWAVE CODE
    aligned_waveforms = align_waveforms(waveforms) #NOT NEEDED
    sqi4_values = []
    
    # Analyze each waveform to determine the proportion of non-clipped data points
    for waveform in aligned_waveforms:
        # Calculate differences between consecutive points in the waveform
        diff = np.diff(waveform)
        # Determine where the absolute difference is below the threshold, indicating no clipping
        clipped = np.where((np.abs(diff) >= threshold), 0, 1)
        sqi4_values.append(np.mean(clipped))
    return np.mean(sqi4_values) if sqi4_values else 0



# ADD f1 and f2 kwargs
def Relative_Power_SQI(arr, fs):
    """
    Calculate the Relative Power Signal Quality Index (RSQI) for a given signal.
    RSQI is computed as the ratio of power within a specific frequency band of interest
    to the power within a broader frequency range, using Welch's method for Power Spectral
    Density (PSD) estimation.

    Parameters:
    - arr (numpy array): The input signal from which to calculate RSQI.
    - fs (int): The sampling frequency of the signal.

    Returns:
    - float: The calculated RSQI, representing the ratio of power in the target band to the broader band.
    """
    # Remove the mean from the signal to focus on fluctuations
    arr = arr - np.mean(arr)
    
    # Calculate the PSD using Welch's method
    f, pxx = welch(arr, fs, nperseg=len(arr)//2, noverlap=len(arr)//4)
    
    # Define frequency bands of interest
    f1 = [1, 2.25]  # Target frequency band
    f2 = [0, 8]     # Broader frequency band for comparison
    
    # Calculate the relative power in the frequency bands
    power_f1 = np.trapz(pxx[(f >= f1[0]) & (f <= f1[1])], f[(f >= f1[0]) & (f <= f1[1])])
    power_f2 = np.trapz(pxx[(f >= f2[0]) & (f <= f2[1])], f[(f >= f2[0]) & (f <= f2[1])])
    
    rsqi = power_f1 / power_f2
    return rsqi




def Zero_Cross_Rate(arr):
    """
    Calculate the Zero Crossing Rate of an array. The zero crossing rate is the rate at which the array changes
    from positive to negative or back. 
    
    Parameters:
    - arr (numpy array): Input array.

    Returns:
    - float: Zero Crossing Rate.
    """
    # Initialize a variable to count the number of zero crossings
    zero_crossing_num = 0

    # Loop through the array to count the zero crossings
    for i in range(len(arr) - 1):
        if arr[i] * arr[i + 1] < 0:
            zero_crossing_num += 1

    # Calculate the Zero Crossing Rate (Z_SQI)
    z_sqiz = zero_crossing_num / len(arr)

    return z_sqiz



def Shannon_Entropy(arr):
    """
    Calculate the Shannon entropy of an array. Shannon entropy is a measure of the information
    contained in a signal and its predictability.

    Parameters:
    - arr (numpy array): Input array.

    Returns:
    - float: Shannon entropy of the array.
    """
    # Calculate the squared values of the array
    squared_signal = [x ** 2 for x in arr] #MAKES NO SENSE, same as arr**2
    
    # Calculate the entropy using the formula
    entropy_index = -sum(squared_signal[i] * math.log(squared_signal[i], 2) for i in range(len(arr)))
    return entropy_index



def Std_of_PSD(arr, fs):
    """
    Calculate the standard deviation of the Power Spectral Density (PSD) of an array using Welch's method.

    Parameters:
    - arr: The input array.
    - fs: The sampling frequency of the array. Default is 1.0.

    Returns:
    - std_psd: The standard deviation of the PSD.
    """
    # Compute the Power Spectral Density (PSD) using Welch's method
    frequencies, psd_values = welch(arr, fs) 
    std_psd = np.std(psd_values)
    return std_psd


#SEEMS COMPLICATED, UNLESS GOOD RESULTS, LETS REMOVE
def Dynamic_Correlation_Index(arr, locs, fs):
    """
    Calculate the dynamic correlation index for waveforms extracted around specified peak locations in an array.
    This index measures the consistency of each new pulse waveform against a set of reference waveforms,
    using correlation to evaluate waveform similarity. It dynamically updates reference waveforms and 
    incorporates linear extrapolation to refine the quality assessment.

    Parameters:
    - arr: The input array from which waveforms are extracted.
    - locs: Indices of peaks in the signal to extract waveforms.
    - fs: Sampling frequency of the array.

    Returns:
    - float: Average quality index across all analyzed waveforms, based on their correlation to reference waveforms.
    """
    
    # Step 1: Initialize storage for reference pulses and quality indices
    reference_pulses = []  # To store previous pulses for correlation
    max_reference_pulses = 10  # Maximum number of reference pulses to keep
    quality_indices = []  # To store the quality indices for each pulse
    
    # Step 2: Iterate over each peak location to process waveforms
    for loc in locs:
        pulse_waveforms = extract_waveforms(arr, [loc], fs)
        if not pulse_waveforms:
            continue  # Skip if no waveform is extracted
        
        # Step 3: Center the pulse waveform around its peak
        pulse_waveform = pulse_waveforms[0]  # Get the extracted pulse waveform
        peak_index = np.argmax(pulse_waveform)
        centered_pulse_waveform = np.roll(pulse_waveform, -peak_index + int(len(pulse_waveform) / 2))
        
        # Step 4: Calculate correlation coefficients with reference pulses
        correlations = []
        for reference_pulse in reference_pulses:
            correlation = np.corrcoef(centered_pulse_waveform, reference_pulse)[0, 1]
            correlations.append(correlation)
        
        # Step 5: Determine the quality index based on the maximum correlation found
        if not correlations or max(correlations) < 0.99:
            quality_indices.append(1.0)  # No correlations or correlations below threshold
        else:
            # Calculate correlation with a linearly extrapolated version of the reference pulse
            slope, intercept, _, _, _ = linregress(range(len(centered_pulse_waveform)), centered_pulse_waveform)
            extrapolated_reference = slope * np.arange(len(centered_pulse_waveform)) + intercept
            correlation = np.corrcoef(centered_pulse_waveform, extrapolated_reference)[0, 1]
            quality_indices.append(correlation)
        
        # Step 6: Manage the collection of reference pulses
        if len(reference_pulses) >= max_reference_pulses:
            reference_pulses.pop(0)  # Remove the oldest reference pulse
        reference_pulses.append(centered_pulse_waveform)
    
    # Step 7: Calculate and return the average quality index
    average_quality = np.mean(quality_indices) if quality_indices else 0  
    return average_quality



























# TEMPLATE MATCHING BASED ALGOS, ADD EUCLIDEAN DISTANCE, ADD POSSIBLITY FOR LINEAR SAMPLING
def Euclidean_Distance(arr, locs, fs):
    """
    Compute the average Euclidean distance of waveforms from their mean waveform.
    
    Parameters:
    - arr: Input signal array.
    - locs: Indices of peaks around which waveforms are extracted.
    - fs: Sampling frequency of the signal.
    
    Returns:
    - Average Euclidean distance between each waveform and the reference waveform.
    """
    waveforms = extract_waveforms(arr, locs, fs) #TAKE FROM VITAL WAVE
    if not waveforms:
        return None  # Handle case where no waveforms are extracted
    aligned_waveforms = align_waveforms(waveforms) # NOT NEEDED
    reference_pulse = np.mean(aligned_waveforms, axis=0)
    euclidean_distances = [np.linalg.norm(waveform - reference_pulse) for waveform in aligned_waveforms]
    return np.mean(euclidean_distances)

def Linear_Resampling(arr, locs, fs):
    """
    Compute a Signal Quality Index (SQI) based on linear resampling and correlation of waveforms.
    
    Parameters:
    - arr (numpy array): The signal array from which waveforms are extracted.
    - locs (list of ints): Indices of peaks in the signal to extract waveforms.
    - fs (int): Sampling frequency of the signal.
    
    Returns:
    - float: The average correlation coefficient between the resampled waveforms and the template.
    """
    waveforms = extract_waveforms(arr, locs, fs)
    aligned_waveforms = align_waveforms(waveforms)
    template = generate_template(aligned_waveforms)
    
    # Resample waveforms to the length of the template and calculate correlation coefficients
    sqi2_values = [
        np.corrcoef(resample(waveform, len(template)), template)[0, 1]
        for waveform in aligned_waveforms
    ]
    return np.mean(sqi2_values)


def quality_index_for_waveform(arr:np.ndarray, r_peaks:np.ndarray) -> list:
    """
    Compute the correlation coefficient between each individual waveform with the average template.

    Parameters
    ----------
    arr
        The ECG signal.
    r_peaks
        An array containing the indices of detected R-peaks in the ECG signal.

    Returns
    -------
    list
        containing correlation coefficients and p-values for each individual waveform compared to the average template.

    Examples
    --------

    To find the r-peaks of ECG in a given signal:

    .. plot::
       :include-source:

       from vitalwave.example_data import load_biosignal
       import matplotlib.pyplot as plt

       import numpy as np

       limits = [0,2000]
       time, ecg = load_biosignal(type="ECG")

       fs = (1 / np.mean(np.diff(time)))

       from vitalwave.signal_quality import quality_index_for_waveform
       from vitalwave import peak_detectors

       ecg_r_peaks = peak_detectors.ecg_modified_pan_tompkins(ecg, fs)

       l_correlation = quality_index_for_waveform(arr=ecg, r_peaks=ecg_r_peaks, type="waveform")

       fig, ax = plt.subplots()
       ax.boxplot(l_correlation)
       plt.show()

    """
    l_correlations = []

    waveforms, mean_waveform_ecg = extract_waveforms(arr, r_peaks, mode="nn_interval")
    mean_waveform_ecg = mean_waveform_ecg[~np.isnan(mean_waveform_ecg)]

    for waveform in waveforms:
        waveform[np.isnan(waveform)] = 0
        correlation = np.ma.corrcoef(x=waveform, y=mean_waveform_ecg)[0][1]
        l_correlations.append(correlation)

    return l_correlations

def Waveform_Correlation_Index(arr, locs, fs): #SAME AS ALREADY (the one above)
    """
    Calculate the correlation index for waveforms extracted around specified locations in an array.

    Parameters:
    - arr: The input array from which waveforms are extracted.
    - locs: Indices of peaks in the signal to extract waveforms.
    - fs: Sampling frequency of the array.

    Returns:
    - float: Average correlation coefficient between each waveform and the median waveform.
    """
    # Step 1: Extract waveforms and align them
    waveforms = extract_waveforms(arr, locs, fs)
    aligned_waveforms = align_waveforms(waveforms)    
    
    # Step 2: Compute median waveform
    median_waveform = np.mean(aligned_waveforms, axis=0)    
    
    # Step 3: Compute correlation coefficients
    corr_coefs = []
    for waveform in aligned_waveforms:
        # Ensure waveform and median_waveform are the same length
        correlation = np.corrcoef(waveform[:len(median_waveform)], median_waveform)[0, 1]
        corr_coefs.append(correlation)  # Fix: use append to add each coefficient to the list
    
    # Step 4: Average correlation coefficients
    quality_indice = np.mean(corr_coefs) if corr_coefs else 0 
    return quality_indice



