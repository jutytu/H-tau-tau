from prepare import *
import torch
import pandas as pd
from torch import nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


# pliki = {'ggH_unpol_e/ggh_unpol_e.csv',
#          'VBFH_unpol_e/VBFH_unpol_e.csv',
#          'ggH_unpol/ggh_unpol.csv',
#          'VBFH_unpol/VBFH_unpol.csv',
#          'ggH_unpol_d/ggh_unpol_d.csv',
#          'VBFH_unpol_d/VBFH_unpol_d.csv'}

pliki = {'ggH_CPeven_e/ggh_CPeven_e.csv',
         'VBFH_CPeven_e/VBFH_CPeven_e.csv',
         'ggH_CPeven/ggh_CPeven.csv',
         'VBFH_CPeven/VBFH_CPeven.csv',
         'ggH_CPeven_d/ggh_CPeven_d.csv',
         'VBFH_CPeven_d/VBFH_CPeven_d.csv'}

features = ['tau_0_decay_neutral_p4E',
            'tau_0_decay_neutral_p4X',
            'tau_0_decay_neutral_p4Y',
            'tau_0_decay_neutral_p4Z',
            'tau_1_decay_neutral_p4E',
            'tau_1_decay_neutral_p4X',
            'tau_1_decay_neutral_p4Y',
            'tau_1_decay_neutral_p4Z',
            'tau_0_decay_charged_p4E',
            'tau_0_decay_charged_p4X',
            'tau_0_decay_charged_p4Y',
            'tau_0_decay_charged_p4Z',
            'tau_1_decay_charged_p4E',
            'tau_1_decay_charged_p4X',
            'tau_1_decay_charged_p4Y',
            'tau_1_decay_charged_p4Z',
            'tauspinner_HCP_Theta_0',
            'tauspinner_HCP_Theta_10',
            'tauspinner_HCP_Theta_20',
            'tauspinner_HCP_Theta_30',
            'tauspinner_HCP_Theta_40',
            'tauspinner_HCP_Theta_50',
            'tauspinner_HCP_Theta_60',
            'tauspinner_HCP_Theta_70',
            'tauspinner_HCP_Theta_80',
            'tauspinner_HCP_Theta_90',
            'tauspinner_HCP_Theta_100',
            'tauspinner_HCP_Theta_110',
            'tauspinner_HCP_Theta_120',
            'tauspinner_HCP_Theta_130',
            'tauspinner_HCP_Theta_140',
            'tauspinner_HCP_Theta_150',
            'tauspinner_HCP_Theta_160',
            'tauspinner_HCP_Theta_170',
            'phi_star']

dfs = []
n = 0
for i in pliki:
    df = pd.read_csv(i)
    dfs.append(df)
    n += 1
merged_df = pd.concat(dfs, ignore_index=True)

full_df = merged_df[features]
pd.set_option('display.max_columns', None)
full_df.to_csv('full_df', index=False)

def boost(df):
    sum_x = df[['tau_0_decay_neutral_p4X', 'tau_1_decay_neutral_p4X', 'tau_0_decay_charged_p4X',
                'tau_1_decay_charged_p4X']].sum(axis=1)
    sum_y = df[['tau_0_decay_neutral_p4Y', 'tau_1_decay_neutral_p4Y', 'tau_0_decay_charged_p4Y',
                'tau_1_decay_charged_p4Y']].sum(axis=1)
    sum_z = df[['tau_0_decay_neutral_p4Z', 'tau_1_decay_neutral_p4Z', 'tau_0_decay_charged_p4Z',
                'tau_1_decay_charged_p4Z']].sum(axis=1)

    df['tau_0_decay_neutral_p4X'] = df['tau_0_decay_neutral_p4X'] - sum_x / 4
    df['tau_1_decay_neutral_p4X'] = df['tau_1_decay_neutral_p4X'] - sum_x / 4
    df['tau_0_decay_charged_p4X'] = df['tau_0_decay_charged_p4X'] - sum_x / 4
    df['tau_1_decay_charged_p4X'] = df['tau_1_decay_charged_p4X'] - sum_x / 4

    df['tau_0_decay_neutral_p4Y'] = df['tau_0_decay_neutral_p4Y'] - sum_y / 4
    df['tau_1_decay_neutral_p4Y'] = df['tau_1_decay_neutral_p4Y'] - sum_y / 4
    df['tau_0_decay_charged_p4Y'] = df['tau_0_decay_charged_p4Y'] - sum_y / 4
    df['tau_1_decay_charged_p4Y'] = df['tau_1_decay_charged_p4Y'] - sum_y / 4

    df['tau_0_decay_neutral_p4Z'] = df['tau_0_decay_neutral_p4Z'] - sum_z / 4
    df['tau_1_decay_neutral_p4Z'] = df['tau_1_decay_neutral_p4Z'] - sum_z / 4
    df['tau_0_decay_charged_p4Z'] = df['tau_0_decay_charged_p4Z'] - sum_z / 4
    df['tau_1_decay_charged_p4Z'] = df['tau_1_decay_charged_p4Z'] - sum_z / 4

    df['tau_0_decay_neutral_p4E'] = np.sqrt(df['tau_0_decay_neutral_p4X']**2 + df['tau_0_decay_neutral_p4Y']**2 +
                                            df['tau_0_decay_neutral_p4Z']**2)
    df['tau_1_decay_neutral_p4E'] = np.sqrt(df['tau_1_decay_neutral_p4X'] ** 2 + df['tau_1_decay_neutral_p4Y'] ** 2 +
                                            df['tau_1_decay_neutral_p4Z'] ** 2)
    df['tau_0_decay_charged_p4E'] = np.sqrt(df['tau_0_decay_charged_p4X'] ** 2 + df['tau_0_decay_charged_p4Y'] ** 2 +
                                            df['tau_0_decay_charged_p4Z'] ** 2)
    df['tau_1_decay_charged_p4E'] = np.sqrt(df['tau_1_decay_charged_p4X'] ** 2 + df['tau_1_decay_charged_p4Y'] ** 2 +
                                            df['tau_1_decay_charged_p4Z'] ** 2)

    return df

full_df = boost(full_df)


full_df.loc[:, 'hypothesis'] = full_df[['tauspinner_HCP_Theta_0',
            'tauspinner_HCP_Theta_10',
            'tauspinner_HCP_Theta_20',
            'tauspinner_HCP_Theta_30',
            'tauspinner_HCP_Theta_40',
            'tauspinner_HCP_Theta_50',
            'tauspinner_HCP_Theta_60',
            'tauspinner_HCP_Theta_70',
            'tauspinner_HCP_Theta_80',
            'tauspinner_HCP_Theta_90',
            'tauspinner_HCP_Theta_100',
            'tauspinner_HCP_Theta_110',
            'tauspinner_HCP_Theta_120',
            'tauspinner_HCP_Theta_130',
            'tauspinner_HCP_Theta_140',
            'tauspinner_HCP_Theta_150',
            'tauspinner_HCP_Theta_160',
            'tauspinner_HCP_Theta_170']].idxmax(axis=1)


# zboostowane 4-pedy:
X = full_df[['tau_0_decay_neutral_p4E',
            'tau_0_decay_neutral_p4X',
            'tau_0_decay_neutral_p4Y',
            'tau_0_decay_neutral_p4Z',
            'tau_1_decay_neutral_p4E',
            'tau_1_decay_neutral_p4X',
            'tau_1_decay_neutral_p4Y',
            'tau_1_decay_neutral_p4Z',
            'tau_0_decay_charged_p4E',
            'tau_0_decay_charged_p4X',
            'tau_0_decay_charged_p4Y',
            'tau_0_decay_charged_p4Z',
            'tau_1_decay_charged_p4E',
            'tau_1_decay_charged_p4X',
            'tau_1_decay_charged_p4Y',
            'tau_1_decay_charged_p4Z']]
print(len(X))

X = X.to_numpy()
print(len(X))
X = torch.from_numpy(np.float64(X))
print(len(X))
X = X.to(torch.float32)


def hot1(data):
    zero = torch.zeros(18)
    if data['hypothesis'] == 'tauspinner_HCP_Theta_0':
        zero[0] = 1
        return zero
    elif data['hypothesis'] == 'tauspinner_HCP_Theta_10':
        zero[1] = 1
        return zero
    elif data['hypothesis'] == 'tauspinner_HCP_Theta_20':
        zero[2] = 1
        return zero
    elif data['hypothesis'] == 'tauspinner_HCP_Theta_30':
        zero[3] = 1
        return zero
    elif data['hypothesis'] == 'tauspinner_HCP_Theta_40':
        zero[4] = 1
        return zero
    elif data['hypothesis'] == 'tauspinner_HCP_Theta_50':
        zero[5] = 1
        return zero
    elif data['hypothesis'] == 'tauspinner_HCP_Theta_60':
        zero[6] = 1
        return zero
    elif data['hypothesis'] == 'tauspinner_HCP_Theta_70':
        zero[7] = 1
        return zero
    elif data['hypothesis'] == 'tauspinner_HCP_Theta_80':
        zero[8] = 1
        return zero
    elif data['hypothesis'] == 'tauspinner_HCP_Theta_90':
        zero[9] = 1
        return zero
    elif data['hypothesis'] == 'tauspinner_HCP_Theta_100':
        zero[10] = 1
        return zero
    elif data['hypothesis'] == 'tauspinner_HCP_Theta_110':
        zero[11] = 1
        return zero
    elif data['hypothesis'] == 'tauspinner_HCP_Theta_120':
        zero[12] = 1
        return zero
    elif data['hypothesis'] == 'tauspinner_HCP_Theta_130':
        zero[13] = 1
        return zero
    elif data['hypothesis'] == 'tauspinner_HCP_Theta_140':
        zero[14] = 1
        return zero
    elif data['hypothesis'] == 'tauspinner_HCP_Theta_150':
        zero[15] = 1
        return zero
    elif data['hypothesis'] == 'tauspinner_HCP_Theta_160':
        zero[16] = 1
        return zero
    elif data['hypothesis'] == 'tauspinner_HCP_Theta_170':
        zero[17] = 1
        return zero

full_df['y'] = full_df.apply(hot1, axis=1)
y = torch.stack(full_df['y'].values.tolist())
y = y.to(torch.float32)
y = y.to(torch.float32)


# X - 4-pedy dla 4 decay products
# y - zakodowane jedynka ktora hipoteza najprawdopodobniejsza dla danego eventu

print(len(full_df))
torch.save(X, 'X.pt')
print(len(X))
print(len(y))
torch.save(y, 'y.pt')