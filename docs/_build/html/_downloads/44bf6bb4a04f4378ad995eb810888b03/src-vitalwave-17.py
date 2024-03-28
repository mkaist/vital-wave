from vitalwave.example_data import load_biosignal
import matplotlib.pyplot as plt

import numpy as np

limits = [0,2000]
time, ppg = load_biosignal(type="PPG")

fs = (1 / np.mean(np.diff(time)))

from src.vitalwave.experimental import derive_ppg_signal_peaks
from vitalwave.peak_detectors import msptd

ppg_signal = ppg

ppg_peaks, ppg_feets = msptd(ppg, fs=fs)

heights_peaks = ppg[ppg_peaks]
heights_feets = ppg[ppg_feets]

plt.figure(figsize=(10, 5))
plt.plot(ppg)

plt.scatter(ppg_peaks, heights_peaks, marker='o', color='green')
plt.scatter(ppg_feets, heights_feets, marker='o', color='red')

plt.show()

nd_ppg_with_peaks, nd_ppg_with_peak_types = derive_ppg_signal_peaks(arr=ppg,
                                                                    ppg_peaks=ppg_peaks,
                                                                    ppg_feets=ppg_feets,
                                                                    window_length=9,
                                                                    polyorder=5)

plt.figure(figsize=(10, 5))
plt.plot(ppg)

ecg_peaks = np.where(nd_ppg_with_peaks == 1.0)[0]
heights_peaks = ppg[ecg_peaks]

plt.scatter(ecg_peaks, heights_peaks, marker='o', color='green')
plt.scatter(ppg_feets, heights_feets, marker='o', color='red')

plt.show()