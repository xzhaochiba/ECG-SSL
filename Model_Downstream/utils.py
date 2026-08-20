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



def plot_result_6(result_data, save_path):
    num = np.arange(1, result_data.shape[1]+1)

    plt.figure(figsize=(6, 4))

    plt.plot(num, result_data[0, :], 'black',  linewidth=1.6, label='Loss')
    plt.plot(num, result_data[1, :], 'red',    linewidth=1.6, label='Acc')
    plt.plot(num, result_data[2, :], 'blue',   linewidth=1.6, label='Pre')
    plt.plot(num, result_data[3, :], 'green',  linewidth=1.6, label='Rec')
    plt.plot(num, result_data[4, :], 'brown',  linewidth=1.6, label='Spe')
    plt.plot(num, result_data[5, :], 'purple', linewidth=1.6, label='F1')

    plt.ylim([0, 105])
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()    



def extract_ele(in_list, ext_ele):

    l = in_list.copy()
    for x in l.copy():
        if ext_ele not in x:
            l.remove(x)
    return l



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



def evaluation_2_class(targets_total, predicted_total):    
    tn, fp, fn, tp = metrics.confusion_matrix(targets_total.cpu(),
                                              predicted_total.cpu(), 
                                              labels=[0, 1]).ravel()

    acc = (tn+tp)/(tn+fp+fn+tp)
    pre = tp/(tp+fp)
    rec = tp/(tp+fn)
    spe = tn/(tn+fp)
    f1  = (2*tp)/(2*tp+fp+fn)

    return tn, fp, fn, tp, acc*100., pre*100., rec*100., spe*100., f1*100.

