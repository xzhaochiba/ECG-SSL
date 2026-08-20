#
# SSL Model.
#

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable



class PositionalEncoding(nn.Module):
    '''
    Implement the PE function.
    '''
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add position encodings to embeddings
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)

        return self.dropout(x)



class Transformer(nn.Module):
    '''
    Transformer encoder processes convolved ECG samples
    Stacks a number of TransformerEncoderLayers
    '''
    def __init__(self, d_model, h, d_ff, num_layers, dropout):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.pe = PositionalEncoding(d_model, dropout=0.1)
        
        encode_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.h, 
            dim_feedforward=self.d_ff, 
            dropout=self.dropout)
        
        self.transformer_encoder = nn.TransformerEncoder(encode_layer, self.num_layers)

    def forward(self, x):      
        out = x.permute(0, 2, 1)
        out = self.pe(out)
        out = out.permute(1, 0, 2)
        out = self.transformer_encoder(out)
        out = out.permute(1, 0, 2)
        out = out.reshape(out.shape[0], -1)
        
        return out



class Encoder_Model(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_layers, dropout_rate, feature_dim):
        super(Encoder_Model, self).__init__()
        
        self.f = nn.Sequential(
            nn.Conv1d(12, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Conv1d(256, d_model, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),

            nn.Conv1d(d_model, d_model, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),

            nn.Conv1d(d_model, d_model, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),

            Transformer(d_model, nhead, d_ff, num_layers, dropout=dropout_rate)
            )

        self.g = nn.Sequential(
            nn.Linear(26112, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, feature_dim, bias=True)
            )

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)

        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)



class Downstream_Model(nn.Module):
    def __init__(self, num_class, pretrained_path, feature_dim, d_model, nhead, d_ff, num_layers, dropout_rate):
        super(Downstream_Model, self).__init__()

        # Encoder
        self.f = Encoder_Model(d_model, nhead, d_ff, num_layers, dropout_rate, feature_dim).f


        self.dat_encoder = nn.Sequential(
            nn.Linear(26112, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),

            nn.Linear(8192, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),

            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 64))


        self.age_encoder = nn.Sequential(
            nn.BatchNorm1d(1))


        self.sex_encoder = nn.Sequential(
            nn.BatchNorm1d(1))

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(66, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Linear(16, num_class, bias=True)
            )

        self.load_state_dict(
            torch.load(pretrained_path, map_location='cpu'), 
            strict=False)

    def forward(self, x, y, z):
        x = self.f(x)

        x = self.dat_encoder(x)
        y = self.age_encoder(y)
        z = self.sex_encoder(z)

        data_all = torch.cat((x, y, z), dim=1)

        out = self.fc(data_all)

        return out
