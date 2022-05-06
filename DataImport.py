import sys
import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

debug = False


class OptionDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]
        if self.transform:
            row = self.transform(row)
        return row


def get_device():
    if torch.cuda.is_available() and not debug:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')  # don't have GPU
    return device


def init_weights(m):
    print(m)

def import_data(input_file, device='cpu'):
    input_df = np.array((list(csv.reader(open(input_file, "r"))))[1:], dtype=object)

    size_train = len(input_df)

    ml_th = 0.52
    tp = 0
    tn = 0
    ps = 0

    min_norm = 10000.0
    max_norm = -10000.0

    for i in range(len(input_df)):
        for j in range(4, 23):
            input_df[i][j] = float(input_df[i][j])
        # label
        real0 = (input_df[i][6] + input_df[i][9]) / 2
        label = 1 if input_df[i][17] >= real0 - 0.000000001 else 0

        forecast = 1 if input_df[i][15] >= real0 - 0.000000001 else 0
        input_df[i][3] = forecast
        input_df[i][1] = input_df[i][17] / real0 - 1.0  # RoI
        input_df[i][17] = label
        ps += forecast

        if forecast == 1 and label == 1:
            tp += 1
        elif forecast == 0 and label == 0:
            tn += 1
        # normalization
        # print(input_df[i][0])
        strike = float(input_df[i][0].split(' ')[-2][1:])
        s_ask = input_df[i][10] - strike
        s_bid = input_df[i][11] - strike
        s = (input_df[i][10] + input_df[i][11]) / 2
        norm = (s - strike) / (s + strike)
        if norm > max_norm:
            max_norm = norm
        if norm < min_norm:
            min_norm = norm
        input_df[i][2] = norm

        total = 0.0
        for j in range(4, 10):
            total += input_df[i][j]
        av = total / 6.0
        sigma = 0.0
        for j in range(4, 10):
            sigma += (input_df[i][j] - av) ** 2
        sigma = math.sqrt(sigma / 5.0)
        for j in range(4, 10):
            input_df[i][j] = (input_df[i][j] - av) / sigma
        input_df[i][10] = (s_ask - av) / sigma
        input_df[i][11] = (s_bid - av) / sigma
        input_df[i][15] = (input_df[i][15] - av) / sigma
        input_df[i][16] = (input_df[i][16] - av) / sigma
        # print(input_df[i])

    x = torch.Tensor(input_df[0:size_train, 4:17].astype(float)).float().to(device)
    # print(x)
    x_shape = x.shape
    x = x.reshape((1, x_shape[0], x_shape[1]))
    # print(x)
    y = torch.from_numpy(input_df[0:size_train, 17:18].astype(float)).float().to(device)

    return x, y

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: python3 {sys.argv[0]} file [model]')
        sys.exit(-1)

    input_file = sys.argv[1]
    x, y = import_data(input_file)
    print(x.shape)  # torch.Size([70322, 13]), torch.Size([1, 6, 13])
    print(y.shape)  # torch.Size([70322, 1]), torch.Size([6, 1])
    print(y[:10])

    exit(0)