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