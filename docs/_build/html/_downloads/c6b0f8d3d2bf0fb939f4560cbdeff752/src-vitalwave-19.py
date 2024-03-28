from vitalwave.example_data import load_biosignal
import matplotlib.pyplot as plt

import numpy as np

limits = [0,2000]
time, ecg = load_biosignal(type="ECG")

fs = (1 / np.mean(np.diff(time)))

from vitalwave import peak_detectors
from vitalwave.experimental import get_ecg_signal_peaks

ecg_r_peaks = peak_detectors.ecg_modified_pan_tompkins(ecg, fs)
heights_peaks = ecg[ecg_r_peaks]

plt.figure(figsize=(10, 5))
plt.plot(ecg)
plt.scatter(ecg_r_peaks, heights_peaks, marker='o', color='green')

plt.show()

nd_ecg_with_peaks, nd_ecg_with_peak_types = get_ecg_signal_peaks(arr=ecg, r_peaks=ecg_r_peaks,
                                                                 fs=fs)

plt.figure(figsize=(10, 5))
plt.plot(ecg)

ecg_peaks = np.where(nd_ecg_with_peaks == 1.0)[0]
heights_peaks = ecg[ecg_peaks]

plt.scatter(ecg_peaks, heights_peaks, marker='o', color='green')
plt.show()