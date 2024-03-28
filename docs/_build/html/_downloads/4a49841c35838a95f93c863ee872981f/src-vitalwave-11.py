from vitalwave.example_data import load_biosignal
import matplotlib.pyplot as plt

import numpy as np

limits = [0,2000]
time, ecg = load_biosignal(type="ECG")

fs = (1 / np.mean(np.diff(time)))

from src.vitalwave.features import get_egc_interval_p_t
from src.vitalwave import peak_detectors
from src.vitalwave.experimental import get_ecg_signal_peaks

ecg_r_peaks = peak_detectors.ecg_modified_pan_tompkins(ecg, fs)

nd_ecg_with_peaks, nd_ecg_with_peak_types = get_ecg_signal_peaks(arr=ecg, r_peaks=ecg_r_peaks,
                                                                 fs=fs)

egc_p_peaks = np.asarray(np.where(nd_ecg_with_peak_types == 1.0))[0]
egc_t_peaks = np.asarray(np.where(nd_ecg_with_peak_types == 5.0))[0]

interval = get_egc_interval_p_t(arr=ecg, p_points=egc_p_peaks, t_points=egc_t_peaks)

fig, ax = plt.subplots()
ax.boxplot(interval)
plt.show()