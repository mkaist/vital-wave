from vitalwave.example_data import load_biosignal
import matplotlib.pyplot as plt

import numpy as np

limits = [0,2000]
time, ppg = load_biosignal(type="PPG")

fs = (1 / np.mean(np.diff(time)))

from vitalwave.features import get_ppg_peak_amplitude_ratio
from vitalwave.peak_detectors import msptd

from src.vitalwave.experimental import derive_ppg_signal_peaks

ppg_peaks, ppg_feets = msptd(ppg, fs=fs)

nd_ppg_with_peaks, nd_ppg_with_peak_types = derive_ppg_signal_peaks(arr=ppg,
                                                                    ppg_peaks=ppg_peaks, ppg_feets=ppg_feets,
                                                                    window_length=9, polyorder=5)

ppg_feets = np.asarray(np.where(nd_ppg_with_peak_types == 1.0))[0]
ppg_systolic = np.asarray(np.where(nd_ppg_with_peak_types == 2.0))[0]
ppg_diastolic = np.asarray(np.where(nd_ppg_with_peak_types == 4.0))[0]

l_ratio = get_ppg_peak_amplitude_ratio(arr=ppg, feets=ppg_feets, systolic_peaks=ppg_systolic,
                                       diastolic_peaks=ppg_diastolic)

fig, ax = plt.subplots()
ax.boxplot(l_ratio)
plt.show()