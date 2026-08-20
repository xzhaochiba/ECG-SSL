#
# SSL Transformer Encoder.
#

import os
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from thop import profile, clever_format

import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from model import Encoder_Model
from utils import del_hidden, find_all_suffix, seed_everything, plot_result_loss, del_ele
from data_loader import DatasetECG_Encoder





def train(epoch):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(train_loader)

    for pos_1, pos_2 in train_bar:
        pos_1, pos_2 = pos_1.to(device, non_blocking=True), pos_2.to(device, non_blocking=True)

        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)

        out = torch.cat([out_1, out_2], dim=0)

        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.temperature)
        
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * args.batch_size, device=sim_matrix.device)).bool()

        sim_matrix = sim_matrix.masked_select(mask).view(2 * args.batch_size, -1)

        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.temperature)

        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        total_num += args.batch_size
        total_loss += loss.item() * args.batch_size

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(
            epoch+1, 
            args.epoch, 
            total_loss / total_num)
        )

    return total_loss / total_num





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SSL Stage 1')
    
    parser.add_argument('--d_model',      type=int,   default=512,      help='Transformer')
    parser.add_argument('--nhead',        type=int,   default=8,        help='Transformer')
    parser.add_argument('--d_ff',         type=int,   default=2048,     help='Transformer')
    parser.add_argument('--num_layers',   type=int,   default=6,        help='Transformer')
    parser.add_argument('--dropout_rate', type=float, default=0.1,      help='Transformer')

    parser.add_argument('--feature_dim',  type=int,   default=128,      help='Feature dim for latent vector')
    parser.add_argument('--temperature',  type=float, default=0.5,      help='Temperature used in softmax')
    
    parser.add_argument('--da1',          type=str,   default='random', help='DA method')
    parser.add_argument('--da2',          type=str,   default='random', help='DA method')

    parser.add_argument('--batch_size',   type=int,   default=512,      help='Number of batch')
    parser.add_argument('--epoch',        type=int,   default=500,      help='Number of epoch')
    parser.add_argument('--learn_late',   type=float, default=1e-3,     help='Learning late')

    args = parser.parse_args()


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('\n==> Device:', device)


    seed_everything(999)


    if not os.path.exists('result'):
        os.mkdir('result')

    if not os.path.exists('model_save'):
        os.mkdir('model_save')


    result_csv = {'Train Loss': []}


    save_name = 'Encoder_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
        args.d_model,
        args.nhead,
        args.d_ff,
        args.num_layers,
        args.dropout_rate,
        args.feature_dim,
        args.temperature,
        args.da1, 
        args.da2, 
        args.batch_size, 
        args.epoch, 
        args.learn_late
        )


    # Dataset path.
    PATH = './dataset'

    all_list = find_all_suffix(PATH, '.h5')
    all_list = del_hidden(all_list)
    all_list.sort()

    tr_dat = all_list

    tr_seed = list(range(0, len(all_list))) 
    tr_seed = np.array(tr_seed)

    print('\n==> Train Data:', len(tr_dat))

    train_data = DatasetECG_Encoder(
        data=tr_dat, 
        seed=tr_seed, 
        da1=args.da1, 
        da2=args.da2
        )

    train_loader = DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True,
        drop_last=True
        )


    net = Encoder_Model(
        d_model=args.d_model,
        nhead=args.nhead, 
        d_ff=args.d_ff, 
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        feature_dim=args.feature_dim).to(device)
    

    flops, params = profile(net, inputs=(torch.randn(1, 12, 2000).to(device),))
    flops, params = clever_format([flops, params])
    print('\n==> Model Params: {}, FLOPs: {}'.format(params, flops), '\n')
    

    optimizer = optim.Adam(
        net.parameters(), 
        lr=args.learn_late, 
        weight_decay=1e-6
        )


    for epoch in range(args.epoch):
        train_loss = train(epoch)
        result_csv['Train Loss'].append(train_loss)

        pd.DataFrame(data=result_csv, index=range(1, epoch + 2)).to_csv(
            './result/{}.csv'.format(save_name), 
            index_label = 'Epoch'
            )

        torch.save(
            net.state_dict(), 
            './model_save/{}.pth'.format(save_name + '_' + str(epoch+1))
            )

    plot_result_loss(np.array(result_csv['Train Loss']), './result/' + save_name + '.png')


    result_mean = sum(result_csv['Train Loss'][args.epoch-10 :])/10
    print('\n==> Mean Result:', result_mean)

    df = pd.read_csv('./result/{}.csv'.format(save_name))
    df.loc[args.epoch + 1] = ['Mean', result_mean] 
    df.to_csv('./result/{}.csv'.format(save_name), index=False)
