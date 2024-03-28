Healthcare Variables
====================

Created: May 17, 2023 1:46 PM Last edited by: Markus Walden

Vital Wave - measuring the heart
================================

The library is designed primarily to contain help-functions and
utilities to guide the analysis heart related bio-signals such as the
ECG (Electrocardiography) and PPG (Photoplethysmography).

::

   * ECG is a technique that records the electrical activity of the heart. 
   * PPG is a non-invasive method used to detect changes in blood volume within tissues.

The ECG provides information about the heart’s rhythm, rate, and overall
cardiac health. It helps identify abnormal heart rhythms (arrhythmias),
detect heart attacks, evaluate the effects of certain medications, and
assess the overall condition of the heart muscle.

The PPG measures the variation in light absorption caused by the
pulsatile flow of blood. The changes in light absorption are primarily
due to the expansion and contraction of blood vessels caused by the
pulsatile blood flow with each heartbeat. PPG can be used to measure
heart rate, blood oxygen saturation (SpO2), and cardiovascular health.

Template code
-------------

.. code-block:: python

   import os
   import sys
   import numpy as np
   import matplotlib.pyplot as plt

   module_path = os.path.abspath(os.path.join('.'))

   if module_path not in sys.path:
      sys.path.append(module_path)

   data_path = os.path.abspath(os.path.join('.\\example_data'))

   print(data_path)

   # either use numpy directly to load the resource-files
   nd_ecg = np.load(data_path + "\\ecg_filt.npy")
   nd_ppg = np.load(data_path + "\\ppg_filt.npy")

   fs = 200

   # or use built-in method to do so
   from vitalwave.example_data import load_ecg_or_ppg_from_npy
   
   nd_ecg, fs = load_ecg_or_ppg_from_npy(l_columns = None, clean = True, ecg = True)
   nd_ppg, fs = load_ecg_or_ppg_from_npy(l_columns=None, clean=True, ecg=False)

That's all
