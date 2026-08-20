#
# ECG Data Loader for SSL Model (Encoder & Downstream).
#

import h5py
import numpy as np

from torch.utils.data import Dataset



def add_noise(data, seed):
    np.random.seed(seed)
    data_new = data + np.random.normal(0, 0.05, (data.shape[0], data.shape[1]))
    return data_new.astype('float32')

def channel_random(data, seed):
    np.random.seed(seed)
    data_new = np.random.permutation(data)
    return data_new.astype('float32')

def channel_resize(data, seed):
    np.random.seed(seed)
    scale = np.random.rand(data.shape[0])
    scale = 0.9 + scale*0.2
    scale = scale.repeat(data.shape[1]).reshape(data.shape[0], -1)
    data_new = np.multiply(data, scale)
    return data_new.astype('float32')

def time_out(data, seed):
    for i in range(data.shape[0]):
        np.random.seed(seed * (i+1))
        begin = np.random.randint(low=0, high=10, size=1)
        data[i, begin[0]*200 : (begin[0]+1)*200].fill(0)
    return data.astype('float32')

def base_shift(data, seed):
    np.random.seed(seed)
    base = np.random.rand(data.shape[0])
    base = base*0.2 - 0.1
    base = base.repeat(data.shape[1]).reshape(data.shape[0], -1)
    data_new = data + base
    return data_new.astype('float32')



class DatasetECG_Encoder(Dataset):
    def __init__(self, data, seed, da1, da2):
        self.data = data
        self.seed = seed
        self.da1  = da1
        self.da2  = da2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        f = h5py.File(self.data[index], 'r')
        data = f['data'][()]
        f.close()

        if self.da1 == 'channel_random':
            data_1 = channel_random(data, self.seed[index])
        elif self.da1 == 'add_noise':
            data_1 = add_noise(data, self.seed[index])
        elif self.da1 == 'channel_resize':
            data_1 = channel_resize(data, self.seed[index])
        elif self.da1 == 'time_out':
            data_1 = time_out(data, self.seed[index])
        elif self.da1 == 'base_shift':
            data_1 = base_shift(data, self.seed[index])

        if self.da2 == 'channel_random':
            data_2 = channel_random(data, self.seed[index] *2)
        elif self.da2 == 'add_noise':
            data_2 = add_noise(data, self.seed[index] *2)
        elif self.da2 == 'channel_resize':
            data_2 = channel_resize(data, self.seed[index] *2)
        elif self.da2 == 'time_out':
            data_2 = time_out(data, self.seed[index] *2)
        elif self.da2 == 'base_shift':
            data_2 = base_shift(data, self.seed[index] *2)

        return data_1, data_2



class DatasetECG_Downstream(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        f = h5py.File(self.data[index], 'r')
        dat = f['data'][()]
        f.close()

        dat = dat.astype('float32')

        return dat, self.label[index]


