from vitalwave.example_data import load_biosignal
import matplotlib.pyplot as plt

import numpy as np

time, ecg = load_biosignal(type="ECG")
time, ppg = load_biosignal(type="PPG")

fs = (1 / np.mean(np.diff(time)))

from vitalwave.basic_algos import extract_waveforms
from vitalwave.peak_detectors import ecg_modified_pan_tompkins, msptd

make_odd = lambda x: x + (x % 2 == 0)

# calculate ECG r-peaks
ecg_r_peaks = ecg_modified_pan_tompkins(ecg, fs=fs)

# calculate ppg peaks and valleys - msptd (Modified Smoothed Peak Detection)
ppg_msptd_peaks, ppg_msptd_feet = msptd(ppg, fs=fs)

ppg_wfs, ppg_wfs_mean = extract_waveforms(ppg, ppg_msptd_feet, 'fid_to_fid')
ecg_wfs1, ecg_wfs1_mean = extract_waveforms(ecg, ecg_r_peaks, 'window', int(make_odd(fs)))
ecg_wfs2, ecg_wfs2_mean = extract_waveforms(ecg, ecg_r_peaks, 'nn_interval')

# Plot the waveforms.
def plot_wfs(wfs, wfs_mean, title):
   fig, ax = plt.subplots()

   for wf in wfs:
      ax.plot(wf, c='tab:blue', alpha=0.2)

   ax.plot(wfs_mean, c='tab:orange', linewidth=2)
   ax.set_title(title)
   fig.tight_layout()
   plt.show()

plot_wfs(ppg_wfs, ppg_wfs_mean, 'PPG waveforms, feet to feet')
plot_wfs(ecg_wfs1, ecg_wfs1_mean, 'ECG waveforms, window')
plot_wfs(ecg_wfs2, ecg_wfs2_mean, 'ECG waveforms, NN interval')