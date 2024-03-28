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