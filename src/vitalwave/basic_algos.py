
import numpy as np

from pywt import wavedec, waverec

from scipy.interpolate import interp1d
from scipy.signal import filtfilt, butter, hilbert, sosfiltfilt, windows, medfilt

def butter_filter(arr: np.ndarray, n: int, wn: np.ndarray, filter_type: str, fs: int) -> np.ndarray:
    """
    Performs zero-phase Butterworth filtering.

    Parameters
    ----------
    arr
        Signal that will be filtered.
    n
        Order of the filter.
    wn
        Cutoff frequencies.
    filter_type
        Type of the filter. Alternatives: lowpass, highpass, bandpass, bandstop.
    fs
        Sampling frequency of the signal.
    
    Returns
    -------
    arr_filtered
        Filtered signal.

    Examples
    --------

    Without normalization:

    .. code-block:: python

       basic_algos.butter_filter(arr=nd_ecg, n=4, wn=[0.5, 8], filter_type='bandpass', fs=fs)

    With normalization:

    .. code-block:: python

       basic_algos.min_max_normalize(basic_algos.butter_filter(arr=nd_ecg, n=4, wn=[0.5, 8],
                                                               filter_type='bandpass', fs=fs))

    """
    # Second-order sections.
    sos = butter(n, wn, filter_type, fs=fs, output='sos')
    # Filtering.
    arr_filtered = sosfiltfilt(sos, arr)

    return arr_filtered


def min_max_normalize(arr: np.ndarray, min_val: float=0.0, max_val: float=1.0) -> np.ndarray:
    """
    Min-max normalizes an array.

    Parameters
    ----------
    arr
        Signal which will be normalized.
    min_val
        Minimum value of the resulting signal.
    max_val
        Maximum value of the resulting signal.

    Returns
    -------
    np.ndarray
        Normalized version of arr.

    Examples
    --------

    To normalize the signal-values without using the standard-scaler based method:

    .. code-block:: python

       basic_algos.min_max_normalize(arr = nd_ecg)

    """
    s_norm = min_val + (arr - np.nanmin(arr)) * (max_val - min_val) / \
        (np.nanmax(arr) - np.nanmin(arr))

    return s_norm

def resample(timestamps: np.ndarray, arr: np.ndarray, timestamps_new: np.ndarray=None, dt: float=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample a time series to a new time axis.

    Parameters
    ----------
    timestamps
        Original timestamps.
    arr
        Original values.
    timestamps_new
        Timestamps used as the basis in resampling. This must
        be in the same unit as timestamps.
    dt
        Timestep of the new time series. This must be
        in the same unit as timestamps.
    
    Returns
    timestamps_new
        An array of resampled timestamps.
    arr_new
        An array of resampled values.

    Examples
    --------

    To setup new time based frequency to an existing signal:

    .. code-block:: python

       basic_algos.resample(timestamps = ecg_ts, arr = ecg, ts_new = timestamps_new)

    or by proving the :math:`\Delta` time variable:

    .. code-block:: python

       basic_algos.resample(timestamps = ecg_ts, arr = ecg, dt = 0.005)

    """
    if timestamps_new is None and dt is None:
        raise ValueError('Either timestamps_new or dt must be given.')

    # Array of new timestamps.
    if timestamps_new is None:
        timestamps_new = np.arange(timestamps[0], timestamps[-1] + dt, dt)
    # Interpolation function.
    f_interp = interp1d(timestamps, arr, kind='cubic', fill_value='extrapolate')
    # Resampled timeseries.
    arr_new = f_interp(timestamps_new)

    return timestamps_new, arr_new

def derivative_filter(arr: np.ndarray, fs: int) -> np.ndarray:
    """
    Derivative filtering.

    A derivative filter according to Pan-Tompkins algorithm.

    Parameters
    ----------
    arr
        Data that will be filtered.        
    fs
        Sampling rate.

    Returns
    -------
    arr_filt
        Filtered data.

    Examples
    --------

    To highlight the the peaks and valleys of the original signal.
    The use of the derivative filter is linked with the moving_average_filter:

    .. code-block:: python

       basic_algos.derivative_filter(arr=ecg, fs=200)

    An example to have the functions working together is linked with it.
    """
    # Filter coefficients.
    coeffs = np.array([1, 2, 0, -2, -1]) * (1 / 8) * fs
    # Forward-backward filtering.
    arr_filt = filtfilt(coeffs, 1, arr)

    return arr_filt

def moving_average_filter(arr: np.ndarray, window: int, type = "triang") -> np.ndarray:
    """
    Moving window integration and moving average

    Parameters
    ----------
    arr
        Data that will be integrated.
    window
        Number of samples in the window.
    type
        Valid types to call for are "triang" and "moving_avg"

    Returns
    -------
    nd_array
        Integrated data or moving average data

    Examples
    --------

    To produce a distorted signal highlighting the r-peak in the QRS-complex of the ECG.

    .. plot::
       :include-source:

       import numpy as np

       from vitalwave.basic_algos import moving_average_filter
       from vitalwave.example_data import load_biosignal

       import matplotlib.pyplot as plt

       limits = [0, 2000]

       time, ecg = load_biosignal(type="ECG")
       fs = (1 / np.mean(np.diff(time)))

       nd_arr_triang = moving_average_filter(arr=ecg, window=int(fs * 0.15))

       fig, ax = plt.subplots(1, 1, sharex=True)

       start, stop = limits
       ax.plot(time[start:stop], nd_arr_triang[start:stop])
       ax.set_title('Smoothing')
       ax.set_xlabel('Time [s]')

       fig.tight_layout()

       plt.show()

    The example is linked with the derivative_filter function found in the same module.
    """
    match type:
        case "triang":
            data = np.convolve(arr, windows.triang(window), mode='same')
        case "moving_avg":
            data = np.convolve(arr, np.ones(window), mode='same') / window
        case _:
            data = None

    return data

#NEEDS TO BE CHECKED
def wavelet_transform_signal(arr: np.ndarray, dwt_transform, dlevels, cutoff_low, cutoff_high):
    """
    Designed to work with noisy signal as a first-pass mechanism.

    Performs wavelet decomposition on the input channel using pywt.wavedec,
    This returns a <list> of coefficients. The coefficients in the specified ranges are
    multiplied by zero to remove their contribution.
    Finally, a reconstructed signal is returned based on the wavelet coefficients.

    Parameters
    ----------
    arr
        signal to process
    dwt_transform
        Wavelet transformation function - good defualt: 'bior4.4'
    dlevels
        wavedeck: level parameter
    cutoff_low
        the scale up to which coefficients will be zeroed
    cutoff_high
        the scale from which coefficients will be zeroed

    Returns
    -------
    np.ndarray
        corrected_signal with inverse wavelet transform

    Examples
    --------
    To clean-up noisy signal prior to processing it with the Butterworth bandpass-filter.
    The example-code includes the linking with the Butterworth bandpass filter:

    .. plot::
       :include-source:

       import numpy as np

       from vitalwave.basic_algos import butter_filter, min_max_normalize, wavelet_transform_signal
       from vitalwave.example_data import load_biosignal

       import matplotlib.pyplot as plt

       limits = [0, 1000]

       time, ecg = load_biosignal(type="ECG")
       fs = (1 / np.mean(np.diff(time)))

       nd_ecg_denoiced = wavelet_transform_signal(arr=ecg, dwt_transform='bior4.4', dlevels=9,
                                                  cutoff_low=1, cutoff_high=9)

       ecg_filt_cleaned = min_max_normalize(butter_filter(arr=nd_ecg_denoiced, n=4, wn=[0.5, 8],
                                                          filter_type='bandpass', fs=fs))

       fig, axes = plt.subplots(2, 1, sharex=True)
       start, stop = limits

       axes[0].plot(time[start:stop], ecg[start:stop])
       axes[1].plot(time[start:stop], ecg_filt_cleaned[start:stop])

       axes[0].set_title('Filtered ECG')
       axes[1].set_title('wavedeck ')

       axes[1].set_xlabel('Time [s]')
       fig.tight_layout()

       plt.show()

    """
    coeffs = wavedec(arr, dwt_transform, level=dlevels)

    # scale 0 to cutoff_low
    for ca in range(0, cutoff_low):
        coeffs[ca] = np.multiply(coeffs[ca], [0.0])

    # scale cutoff_high to end
    for ca in range(cutoff_high, len(coeffs)):
        coeffs[ca] = np.multiply(coeffs[ca], [0.0])

    wavelet_trans = waverec(coeffs, dwt_transform)
    return wavelet_trans

def extract_waveforms(arr: np.ndarray, fid_points: np.ndarray, 
                      mode: str, window: int=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts waveforms from a signal using an array of fiducial points.

    Parameters
    ----------
    arr
        Signal from which the waveforms are extracted.
    fid_points
        Fiducial points used as a basis for extracting the waveforms.
    mode
        How the fiducial points are used to extract the waveforms:
        - fid_to_fid: from one fiducial point to the next one.
        For example, from one PPG foot to another one.
        - nn_interval: the waveform is extracted around each
        fiducial point by taking half of the NN interval before
        and after.
        - window: the waveform is extracted around each fiducial
        point using a window. NOTE: In this case the parameter
        window must be defined.
    window
        The number of samples to take around the fiducial points.
        The parameter must be odd. The number of samples taken 
        from left and right is window // 2.

    Returns
    -------
    waveforms
        An array of extracted waveforms where each row corresponds
        to one waveform.
    mean_waveform
        The calculated mean waveform.

    Examples
    --------
    .. plot::
       :include-source:

       from vitalwave.example_data import load_biosignal
       import matplotlib.pyplot as plt

       import numpy as np

       time, ecg = load_biosignal(type="ECG")
       time, ppg = load_biosignal(type="PPG")

       fs = (1 / np.mean(np.diff(time)))

       from vitalwave.basic_algos import extract_waveforms
       from vitalwave.peak_detectors import ecg_modified_pan_tompkins, msptd

       make_odd = lambda x: x + (x % 2 == 0)

       # calculate ECG r-peaks
       ecg_r_peaks = ecg_modified_pan_tompkins(ecg, fs=fs)

       # calculate ppg peaks and valleys - msptd (Modified Smoothed Peak Detection)
       ppg_msptd_peaks, ppg_msptd_feet = msptd(ppg, fs=fs)

       ppg_wfs, ppg_wfs_mean = extract_waveforms(ppg, ppg_msptd_feet, 'fid_to_fid')
       ecg_wfs1, ecg_wfs1_mean = extract_waveforms(ecg, ecg_r_peaks, 'window', int(make_odd(fs)))
       ecg_wfs2, ecg_wfs2_mean = extract_waveforms(ecg, ecg_r_peaks, 'nn_interval')

       # Plot the waveforms.
       def plot_wfs(wfs, wfs_mean, title):
          fig, ax = plt.subplots()

          for wf in wfs:
             ax.plot(wf, c='tab:blue', alpha=0.2)

          ax.plot(wfs_mean, c='tab:orange', linewidth=2)
          ax.set_title(title)
          fig.tight_layout()
          plt.show()

       plot_wfs(ppg_wfs, ppg_wfs_mean, 'PPG waveforms, feet to feet')
       plot_wfs(ecg_wfs1, ecg_wfs1_mean, 'ECG waveforms, window')
       plot_wfs(ecg_wfs2, ecg_wfs2_mean, 'ECG waveforms, NN interval')

    To extract the waveforms from the source-signal Fiducial point is required:
    """
    # Parameter validation.
    if mode == 'window':
        if window is None:
            raise ValueError('Window parameter must be given in window mode.')
        elif window % 2 != 1:
            raise ValueError('Window must be an odd integer.')

    # Max NN interval.
    nn_max = np.max(np.diff(fid_points))
    if mode == 'fid_to_fid':
        # Create an empty array for holding the waveforms.
        waveforms = np.full((len(fid_points) - 1, int(nn_max)), np.nan)
        # Loop through the fiducial points in pairs.
        for i, fds in enumerate(zip(fid_points[:-1], fid_points[1:])):
            waveforms[i, :int(fds[1] - fds[0])] = arr[fds[0]:fds[1]]
    
    elif mode == 'nn_interval':
        # Create an empty array for holding the waveforms.
        waveforms = np.full((len(fid_points) - 2, int(nn_max)), np.nan)
        # Center point of the longest NN interval.
        nn_max_center = nn_max // 2
        # Loop through the fiducial points starting from the second
        # until the second last.
        for i in range(1, len(fid_points) - 1):
            # Number of samples to take from left and right.
            samples_left = (fid_points[i] - fid_points[i - 1]) // 2
            samples_right = (fid_points[i + 1] - fid_points[i]) // 2
            # Place the waveform into the matrix.
            waveforms[i - 1, nn_max_center - samples_left:nn_max_center + samples_right] = \
                arr[fid_points[i] - samples_left:fid_points[i] + samples_right]

        # Remove columns with just NaN values. These columns could happen due to
        # integer divisions used above. This line is just a way to get rid of 
        # nanmean's "Mean of empty slice" warning.
        waveforms = waveforms[:, ~np.isnan(waveforms).all(axis=0)]    

    elif mode == 'window':
        # Create an empty array for holding the waveforms.
        waveforms = np.full((len(fid_points), window), np.nan)
        # Center point.
        wf_center = window // 2
        # Loop through the fiducial points.
        for i, fp in enumerate(fid_points):
            # Number of samples to take from left and right.
            samples_left = min(fp, wf_center)
            samples_right = min(len(arr) - fp, wf_center + 1)
            # Place the waveform into the matrix.
            waveforms[i, wf_center - samples_left:wf_center + samples_right] = \
                arr[fid_points[i] - samples_left:fid_points[i] + samples_right]
        
    # Compute the mean waveform.
    mean_waveform = np.nanmean(waveforms, 0)
    
    return waveforms, mean_waveform

def filter_hr(heart_rates: np.ndarray, kernel_size: int=7, 
              hr_max_diff: int=16, hr_min: int=40, hr_max:int=180) -> np.ndarray:
    """
    Filter instantaneous heart rates (HRs) with a median
    filter.

    Parameters
    ----------
    heart_rates
        An array of instantaneous HRs [bpm].
    kernel_size
        Kernel size used in the median filter.
    hr_max_diff
        Maximum allowed HR difference [bpm].
    hr_min
        Lowest allowed HR [bpm].
    hr_max
        Highest allowed HR [bpm].
    
    Returns
    -------
    heart_rates
        An array of filtered instantaneous HRs.

    Examples
    --------

    To calculate the initial heart-beat validity based on an existing set of values.

    .. plot::
       :include-source:

       import numpy as np

       from vitalwave.example_data import load_biosignal
       import matplotlib.pyplot as plt

       limits = [0, 1000]

       time, ecg = load_biosignal(type="ECG")

       fs = (1 / np.mean(np.diff(time)))

       from vitalwave.basic_algos import butter_filter, filter_hr
       from vitalwave import peak_detectors

       ecg_filt = butter_filter(ecg, 4, [0.5, 25], 'bandpass', fs=fs)
       ecg_r_peaks = peak_detectors.ecg_modified_pan_tompkins(ecg_filt, fs=fs)
       heart_rates = 60 / np.diff(time[ecg_r_peaks])
       heart_rates_filt = filter_hr(heart_rates, 11)

       fig, ax = plt.subplots()
       ax.plot(heart_rates)
       ax.plot(heart_rates_filt)
       ax.set_xlabel('Time [s]')
       ax.set_ylabel('Heart rate [bpm]')
       ax.set_ylim(40, 200)
       fig.tight_layout()
       plt.show()

    The results show normal heart-rate variability along with abnormal.

    """
    # Make a copy to avoid modifying the original array.
    heart_rates = np.copy(heart_rates)
    # Median filtering.
    heart_rates_med = medfilt(heart_rates, kernel_size)
    # Indices to select.
    idxs = (np.abs(heart_rates - heart_rates_med) <= hr_max_diff) & \
        (heart_rates_med >= hr_min) & (heart_rates_med <= hr_max)
    # Mark the rest of the indices to NaNs.
    heart_rates[~idxs] = np.nan

    return heart_rates

import numpy as np
from scipy import signal


#THIS IS ESSENTAILLY SAME AS ABOVE, ONLY SIMPLE sdsd THRS IS ADDED, WE CAN SIMPLY PUT THIS PART INTO ABOVE WITH A BOOLEN ARG, DEFAULT NOT USE
def filter_hr_peaks(peaks, fs=fs, hr_min=40, hr_max=200, kernel_size=7, sdsd_max=0.35):
    """
    Filters peaks detected in PPG or ECG data to identify and exclude unreliable heart rate (HR) readings based on 
    the variability of R-R intervals. It applies a median filter to smooth the HR signal and uses the Standard Deviation
    of Successive Differences (SDSD) to exclude intervals with high variability.

    Parameters:
        peaks (array): Array of detected peaks' indices in the signal.
        fs (int): Sampling frequency of the signal.
        hr_min (int): Minimum allowable heart rate value in BPM.
        hr_max (int): Maximum allowable heart rate value in BPM.
        kernel_size (int): Size of the kernel used for median filtering.
        sdsd_max (float): Maximum allowable SDSD. Peaks resulting in a higher SDSD will be excluded.

    Returns:
        np.array: An array of valid peak indices after filtering.
        float: The mean heart rate computed from the valid R-R intervals.
    """
    
    # Calculate R-R intervals in samples
    rri_s = np.diff(peaks)
    # Convert R-R intervals into seconds
    rri = rri_s / fs
    # Calculate heart rate from R-R intervals
    hr = 60 / rri
    # Apply median filter to the heart rate signal to smooth it
    hr_med = signal.medfilt(hr, kernel_size)
    # Calculate the SDSD for the R-R intervals
    rr_diff = np.diff(rri)
    sdsd = np.std(rr_diff)

    #ONLY THIS SIMPE THRS PART SEEMS TO BE NEW, OTHERWISE ALREADY INCLUDED IN ABOVE FUNCTION 
    # Check if SDSD exceeds the acceptable maximum, if so return empty array and NaN
    if sdsd_max is not None and sdsd > sdsd_max:
        return np.array([]), np.nan
  
    # Create a mask for heart rate values within the specified range
    valid_hr_mask = (hr_med >= hr_min) & (hr_med <= hr_max)
    # Select valid R-R intervals
    valid_rri = rri[valid_hr_mask]
    # Calculate the mean heart rate from valid R-R intervals
    valid_hr_mean = 60 / np.mean(valid_rri)
    # Initialize valid peaks list with the first peak
    valid_peaks = [peaks[0]]
    # Calculate cumulative sum to find the valid peak indices # JUST USE CUMSUM
    cumulative_sum = peaks[0]
    for i in range(len(valid_rri)):
        cumulative_sum += valid_rri[i] * fs
        valid_peaks.append(int(cumulative_sum))
    
    # Return the array of valid peak indices and the mean heart rate
    return np.array(valid_peaks), valid_hr_mean




def homomorphic_hilbert_envelope(arr:np.ndarray, fs:int, order:int = 1, cutoff_fz:int = 8) -> np.ndarray:
    """
    The homomorphic_hilbert_envelope function is applied to enhance the waveform's envelope. steps:

    Parameters
    ----------
    arr
        signal designed for transformation
    fs
        sample frequency
    order
        sharpness of transition between passband and stopband
    cutoff_fz
        The critical frequency or frequencies of the butter-filter

    Returns
    -------
    filtered_envelope
        Filtered envelope of the input signal.

    Examples
    --------

    To calculate the homomorphic Hilbert envelope; in order to produce a low-resolution mock-up signal of the original.

    .. plot::
       :include-source:

       import numpy as np

       from vitalwave.example_data import load_biosignal
       import matplotlib.pyplot as plt

       limits = [0, 1000]

       time, ecg = load_biosignal(type="ECG")

       fs = (1 / np.mean(np.diff(time)))

       from vitalwave.basic_algos import homomorphic_hilbert_envelope

       ecg_hilbert = homomorphic_hilbert_envelope(arr=ecg, fs=fs)

       fig, axes = plt.subplots(2, 1, sharex=True)
       start, stop = limits

       axes[0].plot(time[start:stop], ecg[start:stop])
       axes[1].plot(time[start:stop], ecg_hilbert[start:stop])

       axes[0].set_title('ECG-signal')
       axes[1].set_title('with Hilbert Envelope')

       plt.show()

    The end result is a signal with key features retained from the original signal
    """
    # Apply a zero-phase low-pass Butterworth filter, 1st order, with a cutoff frequency of 8 Hz
    b_low, a_low = butter(N = order, Wn = cutoff_fz, fs=fs, btype='lowpass')

    # Calculate the Hilbert envelope of the input signal
    analytic_signal = hilbert(arr)
    envelope = np.abs(analytic_signal)

    # Apply the low-pass filter to the log of the envelope
    log_envelope = np.log(envelope)
    filtered_envelope = np.exp(filtfilt(b_low, a_low, log_envelope))

    # Remove spurious spikes in the first sample
    filtered_envelope[0] = filtered_envelope[1]

    return filtered_envelope


def calculate_time_delay(arr_ecg:np.ndarray, arr_ppg:np.ndarray, peaks_ecg, fs:int) -> np.ndarray:
    """
    Calculate time delay between ECG and PPG signals based on corresponding peaks.

    Parameters
    ----------
    arr_ecg
        ECG signal.
    arr_ppg
        PPG signal.
    peaks_ecg
        Peaks in the ECG signal.
    fs
        Sampling frequency.
    key
        Identifier, by default "id".

    Returns
    -------
    numpy.ndarray
        Timestamps of corresponding PPG peaks.

    Examples
    --------

    To Syncronize ECG and PPG peaks discovery.

    .. plot::
       :include-source:

       import numpy as np

       from vitalwave.example_data import load_biosignal
       import matplotlib.pyplot as plt

       limits = [0, 1000]

       time, ecg = load_biosignal(type="ECG")
       time, ppg = load_biosignal(type="ECG")

       fs = (1 / np.mean(np.diff(time)))

       from vitalwave.basic_algos import calculate_time_delay
       from vitalwave.peak_detectors import ecg_modified_pan_tompkins, msptd

       # calculate ECG r-peaks
       ecg_r_peaks = ecg_modified_pan_tompkins(ecg, fs=fs)

       # calculate ppg peaks and valleys - msptd (Modified Smoothed Peak Detection)
       ppg_msptd_peaks, ppg_msptd_feet = msptd(ppg, fs=fs)

       locs_ppg = calculate_time_delay(arr_ecg=ecg, arr_ppg=ppg,
                                       peaks_ecg=ecg_r_peaks, fs=int(fs))

       fig, ax = plt.subplots()
       fig.set_size_inches(10, 6)

       ax.plot(time, ppg)

       ax.plot(time[ppg_msptd_feet], ppg[ppg_msptd_feet], 'go')
       ax.plot(time[locs_ppg], ppg[locs_ppg], 'ro')

       ax.set_xlabel('Time [s]')
       ax.set_title("sync ECG with PPG ")

       fig.tight_layout()
       plt.show()

    The results show the systolic PPG-peak discovery with using the ECG-Pan-Tomkins based method.
    """
    locs_ecg_corrected = _find_corresponding(arr_ecg, peaks_ecg, 0.5 * fs)
    locs_ppg = _find_corresponding(arr_ppg, locs_ecg_corrected, 0.5 * fs, sym=False)

    return locs_ppg


def _find_corresponding(arr, peaks, w, sym=True):
    """
    Find corresponding peaks in the given channel using a window of a certain size.

    Parameters
    ----------
    arr
        Signal channel.
    peaks
        List of peaks.
    w : int
        Window size.
    sym : bool, optional
        If True, use a symmetric window; otherwise, use an asymmetric window, by default True.

    Returns
    -------
    corresponding peaks
    """
    lower_w, upper_w = (int(w / 2), int(w / 2)) if sym else (int(0), int(w))
    corr_locs = []
    for loc in peaks:
        l1, l2 = max(loc - lower_w, 0), min(loc + upper_w, len(arr))
        corr_locs.append(l1 + np.argmax(arr[l1:l2]))

    return np.array(corr_locs)


def segmenting (arr:np.ndarray, window_size:int, overlap:int):
    """
    Segment an array into overlapping frames.

    Parameters
    ----------
    arr
        Input array to segment.
    window_size
        Size of the window.
    overlap
        Overlap between segments.

    Returns
    -------
    frames
        List of segmented frames.
    """
    frames = []
    step = window_size - overlap

    for i in range(0, len(arr), step):
        frame = arr[i:i + window_size]
        frames.append(frame)

    return frames


