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