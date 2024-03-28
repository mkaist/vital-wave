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