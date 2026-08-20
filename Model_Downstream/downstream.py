#
# SSL, Downstream.
#

import os
import random
import pickle
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

from tqdm import tqdm

from model import Downstream_Model
from utils import del_hidden, extract_ele, del_ele, find_all_suffix
from utils import seed_everything, evaluation_2_class, plot_result_6
from data_loader import DatasetECG_Downstream



def train(epoch):
    net.train()

    batch_idx, total_loss, total_num = 0, 0.0, 0
    data_bar = tqdm(train_loader)

    for data, age, sex, target in data_bar:
        data, age, sex, target = data.to(device, non_blocking=True), age.to(device, non_blocking=True), sex.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
        out = net(data, age, sex)
            
        loss = loss_criterion(out, target)

        optimizer.zero_grad()
 
        loss.backward()
 
        optimizer.step()

        total_num += data.size(0)
        total_loss += loss.item() * data.size(0)

        _, predicted = out.max(1)

        if batch_idx == 0:
            targets_total = target
            outputs_total = out
            predicted_total = predicted
        else:
            targets_total = torch.cat((targets_total, target))
            outputs_total = torch.cat((outputs_total, out))
            predicted_total = torch.cat((predicted_total, predicted))

        batch_idx += 1


        tn, fp, fn, tp, acc, pre, rec, spe, f1 = evaluation_2_class(targets_total, predicted_total)

        data_bar.set_description('Train [{}/{}] Loss {:>7.4f} | Acc {:>6.2f} | Pre {:>6.2f} | Rec {:>6.2f} | Spe {:>6.2f} | F1 {:>6.2f}'.format(
            epoch+1, 
            args.epoch, 
            total_loss / total_num, 
            acc, pre, rec, spe, f1)
        )


    model_output_train[epoch, :, :] = outputs_total.cpu().detach().numpy()
    targets_total_train[epoch, :]   = targets_total.cpu().detach().numpy()


    return total_loss / total_num, tn, fp, fn, tp, acc, pre, rec, spe, f1



def test(epoch):
    net.eval()

    batch_idx, total_loss, total_num = 0, 0.0, 0
    data_bar = tqdm(test_loader)

    with torch.no_grad():
        for data, age, sex, target in data_bar:
            data, age, sex, target = data.to(device, non_blocking=True), age.to(device, non_blocking=True), sex.to(device, non_blocking=True), target.to(device, non_blocking=True)

            out = net(data, age, sex)
            
            loss = loss_criterion(out, target)

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            
            _, predicted = out.max(1)

            if batch_idx == 0:
                targets_total = target
                outputs_total = out
                predicted_total = predicted
            else:
                targets_total = torch.cat((targets_total, target))
                outputs_total = torch.cat((outputs_total, out))
                predicted_total = torch.cat((predicted_total, predicted))

            batch_idx += 1


            tn, fp, fn, tp, acc, pre, rec, spe, f1 = evaluation_2_class(targets_total, predicted_total)

            data_bar.set_description('Test  [{}/{}] Loss {:>7.4f} | Acc {:>6.2f} | Pre {:>6.2f} | Rec {:>6.2f} | Spe {:>6.2f} | F1 {:>6.2f}'.format(
                epoch+1, 
                args.epoch, 
                total_loss / total_num,
                acc, pre, rec, spe, f1)
            )


        model_output_test[epoch, :, :] = outputs_total.cpu().detach().numpy()
        targets_total_test[epoch, :]  = targets_total.cpu().detach().numpy()


    return total_loss / total_num, tn, fp, fn, tp, acc, pre, rec, spe, f1





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
        
    parser.add_argument('--k_folder',     type=int,   default=0,    help='K Folder')

    parser.add_argument('--d_model',      type=int,   default=512,  help='Transformer')
    parser.add_argument('--nhead',        type=int,   default=8,    help='Transformer')
    parser.add_argument('--d_ff',         type=int,   default=2048, help='Transformer')
    parser.add_argument('--num_layers',   type=int,   default=6,    help='Transformer')
    parser.add_argument('--dropout_rate', type=float, default=0.1,  help='Transformer')

    parser.add_argument('--feature_dim',  type=int,   default=128,  help='Feature dim for latent vector')
    parser.add_argument('--model_path',   type=str,   default='./model_save/SimCLR_128_0.5_200_64_100_0.001.pth', help='The pretrained model path')

    parser.add_argument('--batch_size',   type=int,   default=512,  help='Number of images in each mini-batch')
    parser.add_argument('--epoch',        type=int,   default=400,  help='Number of sweeps over the dataset to train')
    parser.add_argument('--learn_late',   type=float, default=1e-3, help='Learning Late')

    args = parser.parse_args()
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('\n==> Device :', device)


    seed_everything(999999)


    if not os.path.exists('result'):
        os.mkdir('result')

    if not os.path.exists('model_save'):
        os.mkdir('model_save')


    result_csv = {'Train Loss': [], 
                  'Train TN': [], 'Train FP': [], 'Train FN': [], 'Train TP': [], 
                  'Train Acc': [], 'Train Pre': [], 'Train Rec': [], 'Train Spe': [], 'Train F1': [],
                  'Test Loss': [], 
                  'Test TN': [], 'Test FP': [], 'Test FN': [], 'Test TP': [],
                  'Test Acc': [], 'Test Pre': [], 'Test Rec': [], 'Test Spe': [], 'Test F1': []}

    save_name = 'Downstream_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
        args.model_path.split('/')[-1], 
        args.k_folder,
        args.d_model,
        args.nhead,
        args.d_ff,
        args.num_layers,
        args.dropout_rate,
        args.feature_dim,
        args.batch_size, 
        args.epoch, 
        args.learn_late
        )



    # Train & Test Datasets
    all_list_1 = find_all_suffix('../4_Dataset_Downstream/' + str(args.k_folder) + '/Tr_Stem/', '.h5')
    all_list_2 = find_all_suffix('../4_Dataset_Downstream/' + str(args.k_folder) + '/Tr_Stem_Aug/', '.h5')
    all_list_3 = find_all_suffix('../4_Dataset_Downstream/' + str(args.k_folder) + '/Te_Stem/', '.h5')

    all_list_4 = find_all_suffix('../4_Dataset_Downstream/' + str(args.k_folder) + '/Tr_Takotsubo/', '.h5')
    all_list_5 = find_all_suffix('../4_Dataset_Downstream/' + str(args.k_folder) + '/Tr_Takotsubo_Aug/', '.h5')
    all_list_6 = find_all_suffix('../4_Dataset_Downstream/' + str(args.k_folder) + '/Te_Takotsubo/', '.h5')
    
    all_list_1 = del_hidden(all_list_1)
    all_list_2 = del_hidden(all_list_2)
    all_list_3 = del_hidden(all_list_3)
    all_list_4 = del_hidden(all_list_4)
    all_list_5 = del_hidden(all_list_5)
    all_list_6 = del_hidden(all_list_6)
    
    all_list_1.sort()
    all_list_2.sort()
    all_list_3.sort()
    all_list_4.sort()
    all_list_5.sort()
    all_list_6.sort()

    print(len(all_list_1), len(all_list_2), len(all_list_3), len(all_list_4), len(all_list_5), len(all_list_6))

    tr_dat_stem = all_list_1 + all_list_2
    tr_dat_tako = all_list_4 + all_list_5

    tr_dat = tr_dat_stem + tr_dat_tako

    tr_lab = [1]*len(tr_dat_stem) + [0]*len(tr_dat_tako)
    tr_lab = np.array(tr_lab)
    

    te_dat_stem = all_list_3
    te_dat_tako = all_list_6

    te_dat = te_dat_stem + te_dat_tako

    te_lab = [1]*len(te_dat_stem) + [0]*len(te_dat_tako)
    te_lab = np.array(te_lab)


    print('\n==> Train Data:', len(tr_dat))
    print('==> Test  Data:', len(te_dat))


    model_output_train = np.zeros([args.epoch, len(tr_dat), 2])
    model_output_test  = np.zeros([args.epoch, len(te_dat), 2])  

    targets_total_train = np.zeros([args.epoch, len(tr_dat)])
    targets_total_test  = np.zeros([args.epoch, len(te_dat)])

    

    train_set = DatasetECG_Downstream(data=tr_dat, label=tr_lab)

    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=2)


    test_set = DatasetECG_Downstream(data=te_dat, label=te_lab)

    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=2)



    # Build Model.
    net = Downstream_Model(
        num_class=2,
        pretrained_path=args.model_path,
        feature_dim=args.feature_dim,
        d_model=args.d_model, 
        nhead=args.nhead, 
        d_ff=args.d_ff, 
        num_layers=args.num_layers, 
        dropout_rate=args.dropout_rate).to(device)
    
    for param in net.f.parameters():
        param.requires_grad = False



    optimizer = optim.Adam(
        net.fc.parameters(), 
        lr=args.learn_late, 
        weight_decay=1e-6)


    loss_criterion = nn.CrossEntropyLoss()


    best_acc = 0.0

    for epoch in range(args.epoch):
        train_loss, tn, fp, fn, tp, acc, pre, rec, spe, f1 = train(epoch)

        result_csv['Train Loss'].append(train_loss)
        result_csv['Train TN'].append(tn)
        result_csv['Train FP'].append(fp)
        result_csv['Train FN'].append(fn)
        result_csv['Train TP'].append(tp)
        result_csv['Train Acc'].append(acc)
        result_csv['Train Pre'].append(pre)
        result_csv['Train Rec'].append(rec)
        result_csv['Train Spe'].append(spe)
        result_csv['Train F1'].append(f1)

        
        test_loss, tn, fp, fn, tp, acc, pre, rec, spe, f1 = test(epoch)

        result_csv['Test Loss'].append(test_loss)
        result_csv['Test TN'].append(tn)
        result_csv['Test FP'].append(fp)
        result_csv['Test FN'].append(fn)
        result_csv['Test TP'].append(tp)
        result_csv['Test Acc'].append(acc)
        result_csv['Test Pre'].append(pre)
        result_csv['Test Rec'].append(rec)
        result_csv['Test Spe'].append(spe)
        result_csv['Test F1'].append(f1)
        print()


        # Save result.
        pd.DataFrame(data=result_csv, index=range(1, epoch + 2)).to_csv(
            './result/{}.csv'.format(save_name), 
            index_label = 'Epoch'
            )

        if epoch >= args.epoch-1:
            torch.save(
                net.state_dict(), 
                'model_save/{}.pth'.format(save_name + '_' + str(epoch+1))
                )
        else:
            pass
        

    # Plot result.
    result_plot = np.array(
        result_csv['Test Loss'] +\
        result_csv['Test Acc'] +\
        result_csv['Test Pre'] +\
        result_csv['Test Rec'] +\
        result_csv['Test Spe'] +\
        result_csv['Test F1']).reshape(6, -1)

    plot_result_6(result_plot, './result/' + save_name + '.png')


    result_mean = np.zeros([1, 20])

    result_mean[0, 0] = sum(result_csv['Train Loss'][args.epoch-10 :])/10
    result_mean[0, 1] = sum(result_csv['Train TN'][args.epoch-10 :])/10
    result_mean[0, 2] = sum(result_csv['Train FP'][args.epoch-10 :])/10
    result_mean[0, 3] = sum(result_csv['Train FN'][args.epoch-10 :])/10
    result_mean[0, 4] = sum(result_csv['Train TP'][args.epoch-10 :])/10
    result_mean[0, 5] = sum(result_csv['Train Acc'][args.epoch-10 :])/10
    result_mean[0, 6] = sum(result_csv['Train Pre'][args.epoch-10 :])/10
    result_mean[0, 7] = sum(result_csv['Train Rec'][args.epoch-10 :])/10
    result_mean[0, 8] = sum(result_csv['Train Spe'][args.epoch-10 :])/10
    result_mean[0, 9] = sum(result_csv['Train F1'][args.epoch-10 :])/10

    result_mean[0, 10] = sum(result_csv['Test Loss'][args.epoch-10 :])/10
    result_mean[0, 11] = sum(result_csv['Test TN'][args.epoch-10 :])/10
    result_mean[0, 12] = sum(result_csv['Test FP'][args.epoch-10 :])/10
    result_mean[0, 13] = sum(result_csv['Test FN'][args.epoch-10 :])/10
    result_mean[0, 14] = sum(result_csv['Test TP'][args.epoch-10 :])/10
    result_mean[0, 15] = sum(result_csv['Test Acc'][args.epoch-10 :])/10
    result_mean[0, 16] = sum(result_csv['Test Pre'][args.epoch-10 :])/10
    result_mean[0, 17] = sum(result_csv['Test Rec'][args.epoch-10 :])/10
    result_mean[0, 18] = sum(result_csv['Test Spe'][args.epoch-10 :])/10
    result_mean[0, 19] = sum(result_csv['Test F1'][args.epoch-10 :])/10

    print('\n==> Mean Result:', result_mean)

    df = pd.read_csv('./result/{}.csv'.format(save_name))
    df.loc[args.epoch + 1] = ['Mean'] + list(result_mean[0, :])
    df.to_csv('./result/{}.csv'.format(save_name), index=False)


    file_path = open('./result/output_tr_' + save_name + '.pkl', 'wb')
    pickle.dump(model_output_train, file_path)
    file_path.close()

    file_path = open('./result/output_te_' + save_name + '.pkl', 'wb')
    pickle.dump(model_output_test, file_path)
    file_path.close()

    file_path = open('./result/targets_tr_' + save_name + '.pkl', 'wb')
    pickle.dump(targets_total_train, file_path)
    file_path.close()

    file_path = open('./result/targets_te_' + save_name + '.pkl', 'wb')
    pickle.dump(targets_total_test, file_path)
    file_path.close()

