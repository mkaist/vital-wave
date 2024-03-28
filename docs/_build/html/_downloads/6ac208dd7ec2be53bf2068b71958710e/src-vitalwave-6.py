import numpy as np

from vitalwave.basic_algos import butter_filter, min_max_normalize, wavelet_transform_signal
from vitalwave.example_data import load_biosignal

import matplotlib.pyplot as plt

limits = [0, 1000]

time, ecg = load_biosignal(type="ECG")
fs = (1 / np.mean(np.diff(time)))

nd_ecg_denoiced = wavelet_transform_signal(arr=ecg, dwt_transform='bior4.4', dlevels=9,
                                           cutoff_low=1, cutoff_high=9)

ecg_filt_cleaned = min_max_normalize(butter_filter(arr=nd_ecg_denoiced, n=4, wn=[0.5, 8],
                                                   filter_type='bandpass', fs=fs))

fig, axes = plt.subplots(2, 1, sharex=True)
start, stop = limits

axes[0].plot(time[start:stop], ecg[start:stop])
axes[1].plot(time[start:stop], ecg_filt_cleaned[start:stop])

axes[0].set_title('Filtered ECG')
axes[1].set_title('wavedeck ')

axes[1].set_xlabel('Time [s]')
fig.tight_layout()

plt.show()