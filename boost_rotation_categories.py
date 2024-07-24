# Another boost-rotation file (centre-of-mass frame of reference). Rotated four-momenta were used in the analysis: https://arxiv.org/pdf/2001.00455.
# The rotation part has been removed from this file in order to add some variation to what has already been done. This time the boost is performed
# in the non-relativistic way. The boosted data is used as input for a neural network later on.


import torch
import pandas as pd
import numpy as np


pliki = {'ggH_unpol_e/ggh_unpol_e.csv',
         'VBFH_unpol_e/VBFH_unpol_e.csv',
         'ggH_unpol/ggh_unpol.csv',
         'VBFH_unpol/VBFH_unpol.csv',
         'ggH_unpol_d/ggh_unpol_d.csv',
         'VBFH_unpol_d/VBFH_unpol_d.csv'}


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
for i in pliki:  # merging the files together
    df = pd.read_csv(i)
    dfs.append(df)
    n += 1
merged_df = pd.concat(dfs, ignore_index=True)

full_df = merged_df[features]
pd.set_option('display.max_columns', None)
full_df.to_csv('full_df', index=False)

def boost(df):  # non-relativistic boost to the centre-of-mass frame 
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


full_df.loc[:, 'hypothesis'] = full_df[['tauspinner_HCP_Theta_0',  # finding the most probable hypothesis for each event, i.e. the one for which the weight is the biggest
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


#  X - data frame with boosted four-momenta
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
X = X.to(torch.float32) # converting X to a torch tensor, useful for the neural network


def hot1(data):  # one-hot encoding the most probable hypotehsis: vector with 18 elements, all of them = 0 except the one corresponding to the hypothesis (element position = angle/10)
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
y = y.to(torch.float32)  # saving the hypotheses as the pytorch tensor y


print(len(full_df))
torch.save(X, 'X.pt')
print(len(X))
print(len(y))
torch.save(y, 'y.pt') # saving for future use in the neural network
