

import numpy as np

from vitalwave import basic_algos
from scipy.signal import find_peaks, detrend, windows

import copy

#MAKE L = 2*fs ALWAYS IN ALL METHODS
def ampd(arr: np.ndarray, fs: int=None) -> np.ndarray:
    """
    Automatic multiscale-based peak detection (AMPD).

    Based on the article: https://doi.org/10.3390/a5040588.

    Parameters
    ----------
    arr
        The data for which the peaks are detected.
    fs
        Sampling frequency of the data.

    Returns
    -------
    peak_indices
        Indices for the found peaks in the signal.

    Examples
    --------

    To find the peaks or valleys in a given signal:

    .. plot::
       :include-source:

       from vitalwave.example_data import load_biosignal
       import matplotlib.pyplot as plt

       import numpy as np

       limits = [0,2000]
       time, ppg = load_biosignal(type="PPG")

       fs = (1 / np.mean(np.diff(time)))

       from vitalwave.peak_detectors import ampd

       # calculate ppg peaks and valleys - ampd (Automated Multi-scale Peak Detection)
       start, stop = limits

       ppg_ampd_peaks = ampd(ppg[start:stop], fs=int(fs))

       fig, ax = plt.subplots()
       fig.set_size_inches(10, 6)

       ax.plot(time[start:stop], ppg[start:stop])
       ax.plot(time[start:stop][ppg_ampd_peaks], ppg[start:stop][ppg_ampd_peaks], 'go')

       ax.set_xlabel('Time [s]')
       ax.set_title("Automatic multiscale-based peak detection (AMDP); for PPG")

       fig.tight_layout()
       plt.show()
    """
    # Linear detrending, i.e. subtract a linear regression line from the signal.
    arr_detrended = detrend(arr, type='linear')

    # Number of samples in the signal.
    N = arr_detrended.size

    # Number of moving windows.
    if fs is not None:
        L = 2 * fs
    else:
        L = N // 2

    # Local maxima scalogram (LMS) of the signal.
    M = np.zeros((L, N))
    for k in range(1, L):
        M[k - 1][k:N - k] = (arr_detrended[k:N - k] > arr_detrended[0:N - 2 * k]) & (arr_detrended[k:N - k] > arr_detrended[2 * k:N])

    # Calculate the gamma vector containing the information about the scale-dependent
    # distribution of zeros (and thus local maxima).
    g = np.sum(M, axis=1)

    # Calculate the global maximum of gamma, lambda, which represents the scale with the most
    # local maxima.
    l = np.argmax(g)

    # Reshape the LMS matrix by removing all elements for which k > l.
    M_r = M[0:l, :]

    # Take the minimum of M_r column-wise.
    M_r_col_mins = np.min(M_r, axis=0)

    # The peak indices are the ones where the M_r_mins are zero.
    peaks = np.flatnonzero(M_r_col_mins)

    return peaks


def msptd(arr: np.ndarray, fs: int=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Modified automatic multiscale-based peak detection (AMPD).

    Based on the Matlab implementation presented in the 
    article: https://doi.org/10.1007/978-3-319-65798-1_39.

    Parameters
    ----------
    arr
        The data for which the peaks and troughs are detected
    fs
        Sampling frequency of the data.

    Returns
    -------
    peaks
        Indices for the found peaks.
    troughs
        Indices for the found troughs.

    Examples
    --------

    To find the peaks and valleys in a given signal:

    .. plot::
       :include-source:

       from vitalwave.example_data import load_biosignal
       import matplotlib.pyplot as plt

       import numpy as np

       limits = [0,2000]
       time, ppg = load_biosignal(type="PPG")

       fs = (1 / np.mean(np.diff(time)))

       from vitalwave.peak_detectors import msptd

       start, stop = limits

       # calculate ppg peaks and valleys - msptd (Modified Smoothed Peak Detection)
       ppg_msptd_peaks, ppg_msptd_feet = msptd(ppg[start:stop], fs=fs)

       fig, ax = plt.subplots()
       fig.set_size_inches(10, 6)

       ax.plot(time[:stop], ppg[:stop])
       ax.plot(time[:stop][ppg_msptd_peaks], ppg[:stop][ppg_msptd_peaks], 'go')
       ax.plot(time[:stop][ppg_msptd_feet], ppg[:stop][ppg_msptd_feet], 'ro')

       ax.set_xlabel('Time [s]')
       ax.set_title("Modified AMDP Peak Detection; for PPG")

       fig.tight_layout()
       plt.show()
    """
    N = len(arr)
    if fs is not None:
        L = int(np.ceil(fs / 2)) - 1
    else:
        L = int(np.ceil(N / 2)) - 1

    # Detrend the data.
    arr_detrended = copy.deepcopy(arr)
    arr_detrended[np.isnan(arr_detrended)] = np.nanmean(arr_detrended)
    arr_detrended = detrend(arr_detrended, type='linear')

    # Produce the local maxima and minima scalograms.
    scalogram_max = np.zeros((N, L))
    scalogram_min = np.zeros((N, L))
    for j in range(1, L + 1):
        k = j
        for i in range(k + 2, N - k + 1):
            if arr_detrended[i - 1] > arr_detrended[i - k - 1] and arr_detrended[i - 1] > arr_detrended[i + k - 1]:
                scalogram_max[i - 1, j - 1] = 1
            
            if arr_detrended[i - 1] < arr_detrended[i - k - 1] and arr_detrended[i - 1] < arr_detrended[i + k - 1]:
                scalogram_min[i - 1, j - 1] = 1

    # Form y the column-wise count of where Mx is 0, a scale-dependent 
    # distribution of local maxima. Find d, the scale with the most maxima 
    # (== most number of zeros in row). Redimension scalogram_max to contain 
    # only the first d scales.
    y = np.sum(scalogram_max, axis=0)
    d = np.argmax(y)
    scalogram_max = scalogram_max[:, :d]
    # Do the same for the minima.
    y = np.count_nonzero(scalogram_min, axis=0)
    d = np.argmax(y)
    scalogram_min = scalogram_min[:, :d]
    
    # Form z_max and z_min the row-rise counts of scalogram_max and 
    # scalogram_min's non-zero elements. Any row with a zero count contains 
    # entirley zeros, thus indicating%the presence of a peak or trough.
    z_max = np.count_nonzero(scalogram_max == 0, axis=1)
    z_min = np.count_nonzero(scalogram_min == 0, axis=1)
    
    # Find all the zeros in z_max and z_min. The indices of the zero counts
    # correspond to the position of peaks and troughs respectively.
    peaks = np.nonzero(z_max==0)[0]
    troughs = np.nonzero(z_min==0)[0]

    return peaks, troughs


def ecg_modified_pan_tompkins(arr: np.ndarray, fs: int) -> np.ndarray:
    """
    Detects ECG R peaks using the Pan-Tompkins algorithm.

    Parameters
    ----------
    arr
        ECG signal from which the R peaks are detected.
    fs
        ECG signal sampling frequency.
    
    Returns
    -------
    r_e

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

       from vitalwave.peak_detectors import ecg_modified_pan_tompkins

       start, stop = limits

       ecg_r_peaks = ecg_modified_pan_tompkins(ecg[start:stop], fs=fs)

       fig, ax = plt.subplots()
       fig.set_size_inches(10, 6)

       ax.plot(time[:stop], ecg[:stop])
       ax.plot(time[:stop][ecg_r_peaks], ecg[:stop][ecg_r_peaks], 'go')

       ax.set_xlabel('Time [s]')
       ax.set_title("ECG r-peaks using Pan Tompkins")

       fig.tight_layout()
       plt.show()

    """
    # Bandpass filtering.
    s_filt = basic_algos.butter_filter(arr, 4, [5, 15], 'bandpass', fs)

    # Derivative.
    s_der = basic_algos.derivative_filter(arr=s_filt, fs=fs)

    # Squaring.
    s_sqr = np.square(s_der)

    # Moving-window integration.
    s_int = basic_algos.moving_average_filter(arr=s_sqr, window=int(fs * 0.15), type = "triang")

    # Min-max normalize.
    s_norm = basic_algos.min_max_normalize(s_int)

    # Find peaks.
    r_peaks = find_peaks(s_norm, distance=int(0.4 * fs), height=0.1)[0]

    # Fix R peak positions.
    r_peaks = basic_algos._find_corresponding(arr = arr, peaks = r_peaks, w = (0.5 * fs))

    return r_peaks


def get_peaks_from_ppg_using_segments(arr: np.ndarray, fs: float, set_overlap:float, signal_window_triangular: float = 0,
                    segment: int = 3, get_peaks_only:bool=False):
    """
    Extract PPG (Photoplethysmogram) peaks from raw data.

    Parameters
    ----------
    arr
        Raw input signal.
    fs
        Sampling frequency of the signal.
    set_overlap
        Overlap between segments (used when segment > 0).
    signal_window_triangular
        Using convolution for signal processing; add smoothing (default is 0, indicating no convolution).
    segment : optional
        Number of segments to split the signal into (default is 3). Set to 0 for non-segmented processing.
    get_peaks_only:
        used to get peaks only or both as sorted ndarray

    Returns
    -------
    all_peaks : np.ndarray
        Array containing all detected PPG peaks.

    Examples
    --------

    To find the peaks or valleys in a given signal:

    .. plot::
       :include-source:

       from vitalwave.example_data import load_biosignal
       import matplotlib.pyplot as plt

       import numpy as np

       time, ppg = load_biosignal(type="PPG")

       fs = (1 / np.mean(np.diff(time)))

       from vitalwave.peak_detectors import get_peaks_from_ppg_using_segments

       ppg_peaks = get_peaks_from_ppg_using_segments(arr=ppg, fs=int(fs), set_overlap = (fs * 2),
                                                     get_peaks_only=False)

       fig, ax = plt.subplots()
       fig.set_size_inches(10, 6)

       ax.plot(time, ppg)
       ax.plot(time[ppg_peaks], ppg[ppg_peaks], 'go')
       plt.show()

    """
    # to allow for segmented and non segmented signal
    idx_segment_length = 0
    peaks_all = []

    if set_overlap == 0:
        set_overlap = (fs * 2)

    # pre-filter
    ppg_filtered = [basic_algos.min_max_normalize(basic_algos.butter_filter(arr=arr, n=4, wn=[0.5, 8],
                                                                            filter_type='bandpass', fs=int(fs)))]

    # make an area under the curve / volumetric
    if signal_window_triangular > 0:
        ppg_filtered = basic_algos.moving_average_filter(arr=ppg_filtered[0],
                                                         window=int(signal_window_triangular),
                                                         type = "triang")

    # segment the signal / using np.split causes an error unless the split is not even
    if segment > 0:
        total_segment_length = len(ppg_filtered[0]) // segment * segment
        window = (len(ppg_filtered[0]) // segment) + set_overlap

        ppg_filtered = basic_algos.segmenting(arr[:total_segment_length], window_size = int(window), overlap = int(set_overlap))

    # here single segment and multiple segments are processed within the same process.
    for index, ppg_segment in enumerate(ppg_filtered):
        feet_2_ppg = ampd(-ppg_segment, int(fs))
        peaks_2_ppg = ampd(ppg_segment, int(fs))

        initial_peaks_ppg = peaks_2_ppg if get_peaks_only else np.sort(np.concatenate((feet_2_ppg, peaks_2_ppg)))
        initial_peaks_ppg += idx_segment_length

        idx_segment_length += int(ppg_segment.shape[0] - set_overlap)

        peaks_all.append(initial_peaks_ppg)

    # found peaks are returned
    all_peaks = np.unique(np.concatenate(peaks_all))
    return all_peaks



