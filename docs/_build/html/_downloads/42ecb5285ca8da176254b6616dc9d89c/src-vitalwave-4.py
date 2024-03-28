import numpy as np

from vitalwave.example_data import load_biosignal
import matplotlib.pyplot as plt

limits = [0, 1000]

time, ecg = load_biosignal(type="ECG")

fs = (1 / np.mean(np.diff(time)))

from vitalwave.basic_algos import homomorphic_hilbert_envelope

ecg_hilbert = homomorphic_hilbert_envelope(arr=ecg, fs=fs)

fig, axes = plt.subplots(2, 1, sharex=True)
start, stop = limits

axes[0].plot(time[start:stop], ecg[start:stop])
axes[1].plot(time[start:stop], ecg_hilbert[start:stop])

axes[0].set_title('ECG-signal')
axes[1].set_title('with Hilbert Envelope')

plt.show()