import numpy as np

from vitalwave.basic_algos import moving_average_filter
from vitalwave.example_data import load_biosignal

import matplotlib.pyplot as plt

limits = [0, 2000]

time, ecg = load_biosignal(type="ECG")
fs = (1 / np.mean(np.diff(time)))

nd_arr_triang = moving_average_filter(arr=ecg, window=int(fs * 0.15))

fig, ax = plt.subplots(1, 1, sharex=True)

start, stop = limits
ax.plot(time[start:stop], nd_arr_triang[start:stop])
ax.set_title('Smoothing')
ax.set_xlabel('Time [s]')

fig.tight_layout()

plt.show()