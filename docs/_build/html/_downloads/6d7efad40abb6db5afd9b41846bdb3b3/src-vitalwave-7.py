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