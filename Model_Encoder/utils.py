#
# Functions.
#

import os
import torch
import random
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics



def find_all_suffix(path: str, suffix: str, verbose: bool = False) -> list:
    ''' find all files have specific suffix under the path

    :param path: target path
    :param suffix: file suffix. e.g. ".json"/"json"
    :param verbose: whether print the found path
    :return: a list contain all corresponding file path (relative path)
    '''
    result = []
    if not suffix.startswith("."):
        suffix = "." + suffix
    for root, dirs, files in os.walk(path, topdown=False):
        # print(root, dirs, files)
        for file in files:
            if suffix in file:
                file_path = os.path.join(root, file)
                result.append(file_path)
                if verbose:
                    print(file_path)

    return result



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



def plot_result_loss(result_data, save_path):
    num = np.arange(1, result_data.shape[0] + 1)

    plt.figure(figsize=(6, 4))

    plt.plot(num, result_data[:], 'black',  linewidth=1.6)

    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.grid()
    plt.savefig(save_path)
    plt.close()



def del_ele(in_list, ext_ele):

    l = in_list.copy()
    for x in l.copy():
        if ext_ele in x:
            l.remove(x)
    return l



def del_hidden(in_list):
    for x in in_list.copy():
        if x.split('/')[-1][0:2] == '._':
            in_list.remove(x)

    return in_list
