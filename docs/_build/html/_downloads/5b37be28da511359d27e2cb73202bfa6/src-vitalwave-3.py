import numpy as np

from vitalwave.example_data import load_biosignal
import matplotlib.pyplot as plt

limits = [0, 1000]

time, ecg = load_biosignal(type="ECG")

fs = (1 / np.mean(np.diff(time)))

from vitalwave.basic_algos import butter_filter, filter_hr
from vitalwave import peak_detectors

ecg_filt = butter_filter(ecg, 4, [0.5, 25], 'bandpass', fs=fs)
ecg_r_peaks = peak_detectors.ecg_modified_pan_tompkins(ecg_filt, fs=fs)
heart_rates = 60 / np.diff(time[ecg_r_peaks])
heart_rates_filt = filter_hr(heart_rates, 11)

fig, ax = plt.subplots()
ax.plot(heart_rates)
ax.plot(heart_rates_filt)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Heart rate [bpm]')
ax.set_ylim(40, 200)
fig.tight_layout()
plt.show()