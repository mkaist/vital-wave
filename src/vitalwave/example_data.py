import os
import numpy as np

def load_biosignal(l_columns: list = None, type: str = "ECG", clean: bool = True) -> tuple:
    """
    Load ECG (Electrocardiogram) or PPG (Photoplethysmogram) data from numpy files.

    Parameters:
        l_columns (list, optional): A list of column names to include in the returned data. If not provided,
            default column names will be used for the respective data (ECG or PPG).
        type (str, optional): Acceptable values: ECG or PPG.
        clean (bool, optional): If True, load clean data; if False, load data with motion artifacts.
            Default is True.

    Returns:
        tuple: A tuple containing two elements:
            - numpy.ndarray: time
            - numpy.ndarray: data-column
    """
    if type in "ECG":
        filename = 'clean_ecg.npy' if clean else 'motion_ecg.npy'
        if l_columns is None:
            l_columns = ["time", "minMax_ecg", "ecg" ]

    elif type in "PPG":
        filename = 'clean_ppg.npy' if clean else 'motion_ppg.npy'
        if l_columns is None:
            l_columns = ["time", "minMax_ppg_1_ir", "minMax_ppg_1_green"]

    data_path = os.path.join(os.path.dirname(__file__), 'example_data', filename)
    nd_array = np.load(data_path)

    decouple= np.stack([nd_array[field_name].astype(float) for field_name in l_columns], axis=-1)

    time = decouple[:, 0]
    data = decouple[:, 1]

    return time, data

