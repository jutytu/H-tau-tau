# The function 'boost_to_com_frame' boosts the four-momenta of the tau leptons' decay products into the centre-of-mass frame of reference and rotates them so that the tau 
# momenta are placed along the z-axis.

import torch
import pandas as pd
import numpy as np



pliki = {'ggH_unpol_e/ggh_unpol_e.csv',
         'VBFH_unpol_e/VBFH_unpol_e.csv',
         'ggH_unpol/ggh_unpol.csv',
         'VBFH_unpol/VBFH_unpol.csv',
         'ggH_unpol_d/ggh_unpol_d.csv',
         'VBFH_unpol_d/VBFH_unpol_d.csv'}
dfs = []
n = 0

for i in pliki:
    df = pd.read_csv(i)
    dfs.append(df)
    n += 1

merged_df = pd.concat(dfs, ignore_index=True)

dfs2 = []
phi = 0
for i in range(18):
    df = merged_df
    df['alpha_nn'] = phi
    df['weight_nn'] = df[f'tauspinner_HCP_Theta_{phi}']
    dfs2.append(df)
    phi += 10

# creating a data frame with column 'alpha_nn' (the hypothesis angle) and column 'weight_nn' with corresponding weight; every event appears 18 times here, with different weight
full_df = pd.concat(dfs2, ignore_index=True)

#################

#5
features = ['tau_0_p4E',
            'tau_0_p4X',
            'tau_0_p4Y',
            'tau_0_p4Z',
            'tau_1_p4E',
            'tau_1_p4X',
            'tau_1_p4Y',
            'tau_1_p4Z',
            'tau_0_decay_neutral_p4E',
            'tau_0_decay_neutral_p4X',
            'tau_0_decay_neutral_p4Y',
            'tau_0_decay_neutral_p4Z',
            'tau_1_decay_neutral_p4E',
            'tau_1_decay_neutral_p4X',
            'tau_1_decay_neutral_p4Y',
            'tau_1_decay_neutral_p4X',
            'tau_0_decay_charged_p4E',
            'tau_0_decay_charged_p4X',
            'tau_0_decay_charged_p4Y',
            'tau_0_decay_charged_p4X',
            'tau_1_decay_charged_p4E',
            'tau_1_decay_charged_p4X',
            'tau_1_decay_charged_p4Y',
            'tau_1_decay_charged_p4Z',
            'weight_nn',
            'alpha_nn']

df_in = full_df[features]
X = df_in.to_numpy()
y = torch.tensor(full_df['weight_nn'].values, dtype=torch.float32)


def boost_to_com_frame(X):
    def lorentz_boost(p, beta):
        beta2 = np.einsum('ij,ij->i', beta, beta) # using formulas for boost matrix
        gamma = 1 / np.sqrt(1 - beta2)
        bp = np.einsum('ij,ij->i', beta, p[:, 1:])
        p_parallel = beta * bp[:, np.newaxis]
        p_perpendicular = p[:, 1:] - p_parallel
        boosted_p = np.zeros_like(p)
        boosted_p[:, 0] = gamma * (p[:, 0] - bp)
        boosted_p[:, 1:] = p_perpendicular + gamma[:, np.newaxis] * p_parallel + (gamma[:, np.newaxis] - 1) * np.einsum(
            'ij,ij->i', p_parallel, beta)[:, np.newaxis] / beta2[:, np.newaxis]
        return boosted_p

    def compute_beta(p1, p2):
        p_tot = p1 + p2 # calculating the beta matrix
        beta = p_tot[:, 1:] / p_tot[:, 0][:, np.newaxis]
        return beta

    def rotation_matrix(v):
        v = v / np.linalg.norm(v) # normalizing the vector
        x, y, z = v
        cos_theta = z
        sin_theta = np.sqrt(1 - z ** 2)
        cos_phi = x / np.sqrt(x ** 2 + y ** 2)
        sin_phi = y / np.sqrt(x ** 2 + y ** 2)

        if np.isnan(cos_phi): # avoiding NaN values
            cos_phi = 1.0
        if np.isnan(sin_phi):
            sin_phi = 0.0

        # rotation matrix around y-axis to bring z into xz-plane
        R_y = np.array([
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta]
        ])

        # rotation matrix around z-axis to align x-axis with x-component
        R_z = np.array([
            [cos_phi, -sin_phi, 0],
            [sin_phi, cos_phi, 0],
            [0, 0, 1]
        ])

        # combined rotation matrix
        R = R_y @ R_z

        return R

    def rotation_matrices(v):
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        v_normalized = v / norms
        x, y, z = v_normalized[:, 0], v_normalized[:, 1], v_normalized[:, 2]

        cos_theta = z
        sin_theta = np.sqrt(1 - z ** 2)
        cos_phi = np.where(x ** 2 + y ** 2 > 0, x / np.sqrt(x ** 2 + y ** 2), 1.0)
        sin_phi = np.where(x ** 2 + y ** 2 > 0, y / np.sqrt(x ** 2 + y ** 2), 0.0)

        R_y = np.array([
            [cos_theta, np.zeros_like(cos_theta), sin_theta],
            [np.zeros_like(cos_theta), np.ones_like(cos_theta), np.zeros_like(cos_theta)],
            [-sin_theta, np.zeros_like(cos_theta), cos_theta]
        ]).transpose(2, 0, 1)

        R_z = np.array([
            [cos_phi, -sin_phi, np.zeros_like(cos_phi)],
            [sin_phi, cos_phi, np.zeros_like(cos_phi)],
            [np.zeros_like(cos_phi), np.zeros_like(cos_phi), np.ones_like(cos_phi)]
        ]).transpose(2, 0, 1)

        R = R_y @ R_z

        return R

    # rotating the momenta
    def rotate_momenta(tau_p4, decay_p4):
        decay_p3 = decay_p4[:, 1:4]  # getting spatial components of decay products
        tau_p3 = tau_p4[:, 1:4]  # getting spatial components of taus

        R = rotation_matrices(tau_p3)
        rotated_decay_p3 = np.einsum('ijk,ik->ij', R, decay_p3)

        return np.hstack((decay_p4[:, 0:1], rotated_decay_p3))

    # extracting columns
    tau_0_decay_neutral = X[:, 8:12]
    tau_1_decay_neutral = X[:, 12:16]
    tau_0_decay_charged = X[:, 16:20]
    tau_1_decay_charged = X[:, 20:24]
    tau_0 = tau_0_decay_neutral + tau_0_decay_charged
    tau_1 = tau_1_decay_neutral + tau_1_decay_charged
    weight_nn = X[:, 24]
    alpha_nn = X[:, 25]


    # computing the boost vectors for all rows
    beta = compute_beta(tau_0, tau_1)


    # applying Lorentz boost to all four-momenta
    boosted_tau_0 = lorentz_boost(tau_0, beta)
    boosted_tau_1 = lorentz_boost(tau_1, beta)
    boosted_tau_0_decay_neutral = lorentz_boost(tau_0_decay_neutral, beta)
    boosted_tau_1_decay_neutral = lorentz_boost(tau_1_decay_neutral, beta)
    boosted_tau_0_decay_charged = lorentz_boost(tau_0_decay_charged, beta)
    boosted_tau_1_decay_charged = lorentz_boost(tau_1_decay_charged, beta)


    rotated_tau_0_decay_neutral = rotate_momenta(boosted_tau_0, boosted_tau_0_decay_neutral)
    rotated_tau_1_decay_neutral = rotate_momenta(boosted_tau_1, boosted_tau_1_decay_neutral)
    rotated_tau_0_decay_charged = rotate_momenta(boosted_tau_0, boosted_tau_0_decay_charged)
    rotated_tau_1_decay_charged = rotate_momenta(boosted_tau_1, boosted_tau_1_decay_charged)


    boosted_momenta = np.hstack((rotated_tau_0_decay_neutral,
                                 rotated_tau_1_decay_neutral,
                                 rotated_tau_0_decay_charged,
                                 rotated_tau_1_decay_charged))


    result = np.hstack((boosted_momenta, weight_nn[:, np.newaxis], alpha_nn[:, np.newaxis]))

    return result


boosted_rotated_X = boost_to_com_frame(X)
X = torch.from_numpy(boosted_rotated_X)
X = X.to(torch.float32)

print(X)
print(len(X))
print(len(y))


torch.save(X, 'X.pt')
torch.save(y, 'y.pt')
