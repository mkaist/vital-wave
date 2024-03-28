from vitalwave.example_data import load_biosignal
import matplotlib.pyplot as plt

import numpy as np

limits = [0,2000]
time, ppg = load_biosignal(type="PPG")

fs = (1 / np.mean(np.diff(time)))

from vitalwave.experimental import do_ppg_from_raw_signal

ppg_data_all, peak_indices_all = do_ppg_from_raw_signal(arr=ppg, fs=fs, segment=6, threshold=3,
                                                        return_as_segment=False)
heights_peaks_all = ppg_data_all[peak_indices_all]

ppg_data, peak_indices = do_ppg_from_raw_signal(arr=ppg, fs=fs, segment=3, threshold=3,
                                                return_as_segment=True)
heights_peaks = ppg_data[0][peak_indices[0]]

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(ppg_data_all)
axes[0].scatter(peak_indices_all, heights_peaks_all, marker='o', color='green')
axes[0].set_title("PPG Data as complete (Segment = 3)")

axes[1].plot(ppg_data[0])
axes[1].scatter(peak_indices[0], heights_peaks, marker='o', color='green')
axes[1].set_title("PPG Data as segments (Segment = 6)")

plt.tight_layout()

plt.show()