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