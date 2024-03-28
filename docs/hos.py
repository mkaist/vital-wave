import os
import sys

import numpy as np
from matplotlib import pyplot as plt

module_path = os.path.abspath(os.path.join('plot_code'))

if module_path not in sys.path:
    sys.path.append(module_path)

data_path = os.path.abspath(os.path.join('..\\src\\vitalwave\\example_data'))

print(data_path)

nd_ecg = np.load(data_path + "\\ecg_filt.npy")
nd_ppg = np.load(data_path + "\\ppg_filt.npy")

fs = 200
start = 0
stop = 1000
duration = (stop - start) / fs

from vitalwave.peak_detectors import ecg_pan_tompkins, ampd, msptd

# calculate ECG r-peaks
ecg_r_peaks = ecg_pan_tompkins(nd_ecg["filt_ecg_signal"][start:stop], fs=fs)

# calculate ppg peaks and valleys - ampd (Automated Multi-scale Peak Detection)
ppg_ampd_feet = ampd(-nd_ppg["filt_ppg_signal"][start:stop], fs=fs)
ppg_ampd_peaks = ampd(nd_ppg["filt_ppg_signal"][start:stop], fs=fs)

# calculate ppg peaks and valleys - msptd (Modified Smoothed Peak Detection)
ppg_msptd_peaks, ppg_msptd_feet = msptd(nd_ppg["filt_ppg_signal"][start:stop], fs=fs)


from src.collect import show_plot

show_plot(ts_s = nd_ecg["time"][start:stop], s = nd_ecg["filt_ecg_signal"][start:stop], peaks = ecg_r_peaks,
          title = "ECG r-peaks using Pan Tompkins")

show_plot(ts_s = nd_ppg["time"][start:stop], s = nd_ppg["filt_ppg_signal"][start:stop], peaks = ppg_ampd_peaks,
          feet = ppg_ampd_feet, title = "PPG peaks and valleys, using AMPD")

show_plot(ts_s = nd_ppg["time"][start:stop], s = nd_ppg["filt_ppg_signal"][start:stop], peaks = ppg_msptd_peaks,
          feet = ppg_msptd_feet, title = "PPG peaks and valleys, using MSPTD")
