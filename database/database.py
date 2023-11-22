from utils import *
from typing import Tuple


class database():
    def __init__(self, datasetname: str, filename: str):
        """
        初始化datastruct对象。

        参数：
            datasetname (str)：数据集的名称。
            filename (str)：包含数据集的MATLAB文件的文件名。
        """
        self.dataset_name = datasetname
        self.filepath = f'dataset/{filename}.mat'
        self.rawdata = load_mat(self.filepath)
        self.mean = np.ndarray([])
        self.std = np.ndarray([])
        self.slices = []
        self.rawdata = self.__normalize()

    def __normalize(self) -> dict:
        """
        对原始数据进行归一化处理。

        返回：
            dict：包含归一化后数据的字典。
        """
        newdatadict = {}
        for key, value in self.rawdata.items():
            valuemean = np.mean(value, axis=0)
            valuestd = np.std(value, axis=0)
            if not np.any(valuestd == 0):
                newvalue = (value - valuemean) / valuestd
            else:
                newvalue = value
            self.mean = np.append(self.mean, valuemean)
            self.std = np.append(self.std, valuestd)
            newdatadict[key] = newvalue
        return newdatadict

    def rawdatatonumpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return dict_tonp(self.rawdata)

    def discrete(self, slicenum=100, eps=1e-18) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        对数据进行离散化处理。

        参数：
            slicenum (int, 可选)：离散化的分段数。默认为100。
            eps (float, 可选)：避免除零的小值。默认为1e-18。

        返回：
            np.ndarray：离散化后的数据。
            np.ndarray：标签。
            np.ndarray：标签映射。
        """
        self.slicenum = slicenum
        data, label, labelmap = self.rawdatatonumpy()
        if len(data.shape) == 2:
            datamin = np.min(data, axis=0)
            datamax = np.max(data, axis=0)
            datamax = datamax + eps
            assert datamin.shape[0] == data.shape[1]
            assert datamax.shape[0] == data.shape[1]

            num = data.shape[1]
            if not self.slices:
                self.__generateslice(datamax, datamin, num, slicenum)
            for i in range(data.shape[1]):
                data[:, i] = np.digitize(data[:, i], self.slices[i, :])
            data = data.astype(int)
            return data, label, labelmap
        elif len(data.shape) == 3:
            data = data.transpose((0, 2, 1))
            datamin = np.array([np.min(data[:, :, i]) for i in range(data.shape[2])])
            datamax = np.array([np.max(data[:, :, i]) for i in range(data.shape[2])])
            datamax = datamax + eps
            assert datamin.shape[0] == data.shape[2]
            assert datamax.shape[0] == data.shape[2]

            num = data.shape[2]
            if not self.slices:
                self.__generateslice(datamax, datamin, num, slicenum)
            for i in range(data.shape[2]):
                data[:, i, :] = np.digitize(data[:, i, :], self.slices[i, :])
            data: np.ndarray = data.astype(int)
            return data, label, labelmap
        else:
            raise NotImplementedError('len(data.shape)!=2&&len(data.shape)!=3 Not Implement!')

    def __generateslice(self, datamax: np.ndarray, datamin: np.ndarray, num: int, slicenum: int):
        """
        生成离散化的切片。

        参数：
            datamax (np.ndarray)：每个特征的最大值。
            datamin (np.ndarray)：每个特征的最小值。
            num (int)：特征数量。
            slicenum (int)：离散化的分段数。
        """
        for min, max in zip(datamin, datamax):
            label_slice = np.linspace(min, max, slicenum)
            self.slices.append(label_slice)
        self.slices = np.array(self.slices)
        assert self.slices.shape == (num, slicenum)

    def repairRawdata(self, data: np.ndarray, label: np.ndarray, labelmap: np.ndarray):
        rawdata = np_to_dict(data, label, labelmap)
        self.rawdata = rawdata
        self.rawdata = self.__normalize()

