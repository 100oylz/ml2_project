import scipy.io as IO
import numpy as np
from sklearn.model_selection import train_test_split


def load_mat(filepath: str):
    data = IO.loadmat(filepath)
    data_dict = {}
    for key, value in data.items():
        if key not in ('__version__', '__globals__', '__header__'):
            value = np.ascontiguousarray(value)
            data_dict[key] = value.astype('float64')
    return data_dict


def split_train_valid_test(data: np.ndarray, label: np.ndarray, randomstate: int) -> (
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    train_data, temp_data, train_label, temp_label = train_test_split(data, label, test_size=0.4,
                                                                      random_state=randomstate, stratify=label)

    valid_data, test_data, valid_label, test_label = train_test_split(temp_data, temp_label, test_size=0.5,
                                                                      random_state=randomstate, stratify=temp_label)

    return train_data, train_label, valid_data, valid_label, test_data, test_label


def np_to_dict(data, label, labelmap):
    rawdata = {}
    unique_labels = np.unique(label)
    for u_label in unique_labels:
        label_data = data[label == u_label]
        label_name = labelmap[u_label]
        if label_name not in rawdata:
            rawdata[label_name] = label_data.tolist()
        else:
            rawdata[label_name].extend(label_data.tolist())
    return rawdata

def dict_tonp(self, datadict):
    data = []
    label = []
    labelmap = []
    pointer = 0
    for key, value in datadict.items():
        labelmap.append(key)
        for item in value:
            label.append(pointer)
            data.append(item)
        pointer += 1
    return np.array(data), np.array(label), np.array(labelmap)


