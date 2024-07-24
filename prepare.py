# This file prepares the root files used for further analysis. 

# The 'Events' function creates a data frame with a chosen set of branches from the original root file, unpacks the four-vectors ('Vectors4' function) and sets the 
# phi star value according to the decay modes of both taus and their consistency with the matched decay modes ('Conditions function').

# The 'Events2' function does the same thing, but puts extra requirements on the phi star value setting (the number of recorded charged/neutral tracks needs to match 
# the expected number for a given decay mode).


import numpy as np
import uproot
import pandas as pd
import math


def Events(data_file):
    print("Reading file:", data_file)

    file = uproot.open(data_file)
    tree = file['NOMINAL']

    branches = ['tau_0_decay_mode',
                'tau_1_decay_mode',
                'tau_0_matched_decay_mode',
                'tau_1_matched_decay_mode',
                'tau_0_matched_n_charged',
                'tau_1_matched_n_charged',
                'tau_0_matched_n_neutral',
                'tau_1_matched_n_neutral',
                'tau_0_matched_vis_charged_p4',
                'tau_0_matched_vis_neutral_p4',
                'tau_1_matched_vis_charged_p4',
                'tau_1_matched_vis_neutral_p4',
                'ditau_CP_phi_star_cp_a1_rho',
                'ditau_CP_phi_star_cp_ip_ip',
                'ditau_CP_phi_star_cp_ip_rho',
                'ditau_CP_phi_star_cp_ip_rho_opt',
                'ditau_CP_phi_star_cp_rho_ip',
                'ditau_CP_phi_star_cp_rho_rho',
                'ditau_matched_CP_phi_star_cp_a1_rho',
                'ditau_matched_CP_phi_star_cp_ip_ip',
                'ditau_matched_CP_phi_star_cp_ip_rho',
                'ditau_matched_CP_phi_star_cp_rho_ip',
                'ditau_matched_CP_phi_star_cp_rho_rho',
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
                'tauspinner_HCP_Theta_170']

    df = tree.arrays(branches, library='pd')

    # The numbers corresponding to each decay mode can be found in the article: https://arxiv.org/pdf/2212.05833.
    
    def Conditions(data):
        if (data['tau_0_decay_mode'] == 0) and (data['tau_1_decay_mode'] == 0) and \
                (data['tau_0_matched_decay_mode'] == 0) and (data['tau_1_matched_decay_mode'] == 0) and \
                (data['tau_0_matched_n_charged'] == 1) and (data['tau_1_matched_n_charged'] == 1) and \
                (data['tau_0_matched_n_neutral'] == 0) and (data['tau_1_matched_n_neutral'] == 0):
            return data['ditau_matched_CP_phi_star_cp_ip_ip']
        elif (data['tau_0_decay_mode'] == 1) and (data['tau_1_decay_mode'] == 1) and \
                (data['tau_0_matched_decay_mode'] == 1) and (data['tau_1_matched_decay_mode'] == 1) and \
                (data['tau_0_matched_n_charged'] == 1) and (data['tau_1_matched_n_charged'] == 1) and \
                (data['tau_0_matched_n_neutral'] == 1) and (data['tau_1_matched_n_neutral'] == 1):
            return data['ditau_matched_CP_phi_star_cp_rho_rho']
        elif (data['tau_0_decay_mode'] == 3) and (data['tau_1_decay_mode'] == 1) and \
                (data['tau_0_matched_decay_mode'] == 3) and (data['tau_1_matched_decay_mode'] == 1) and \
                (data['tau_0_matched_n_charged'] == 3) and (data['tau_1_matched_n_charged'] == 1) and \
                (data['tau_0_matched_n_neutral'] == 0) and (data['tau_1_matched_n_neutral'] == 1):
            return data['ditau_matched_CP_phi_star_cp_a1_rho']
        elif (data['tau_0_decay_mode'] == 1) and (data['tau_1_decay_mode'] == 3) and \
                (data['tau_0_matched_decay_mode'] == 1) and (data['tau_1_matched_decay_mode'] == 3) and \
                (data['tau_0_matched_n_charged'] == 1) and (data['tau_1_matched_n_charged'] == 3) and \
                (data['tau_0_matched_n_neutral'] == 1) and (data['tau_1_matched_n_neutral'] == 0):
            return data['ditau_matched_CP_phi_star_cp_a1_rho']
        elif (data['tau_0_decay_mode'] == 2) and (data['tau_1_decay_mode'] == 1) and \
                (data['tau_0_matched_decay_mode'] == 2) and (data['tau_1_matched_decay_mode'] == 1) and \
                (data['tau_0_matched_n_charged'] == 1) and (data['tau_1_matched_n_charged'] == 1) and \
                (data['tau_1_matched_n_neutral'] == 1):
            return data['ditau_matched_CP_phi_star_cp_rho_rho']
        elif (data['tau_0_decay_mode'] == 1) and (data['tau_1_decay_mode'] == 2) and \
                (data['tau_0_matched_decay_mode'] == 1) and (data['tau_1_matched_decay_mode'] == 2) and \
                (data['tau_0_matched_n_charged'] == 1) and (data['tau_1_matched_n_charged'] == 1) and \
                (data['tau_0_matched_n_neutral'] == 1):
            return data['ditau_matched_CP_phi_star_cp_rho_rho']
        elif (data['tau_0_decay_mode'] == 0) and (data['tau_1_decay_mode'] == 1) and \
                (data['tau_0_matched_decay_mode'] == 0) and (data['tau_1_matched_decay_mode'] == 1) and \
                (data['tau_0_matched_n_charged'] == 1) and (data['tau_1_matched_n_charged'] == 1) and \
                (data['tau_0_matched_n_neutral'] == 0) and (data['tau_1_matched_n_neutral'] == 1):
            return data['ditau_matched_CP_phi_star_cp_ip_rho']
        elif (data['tau_0_decay_mode'] == 1) and (data['tau_1_decay_mode'] == 0) and \
                (data['tau_0_matched_decay_mode'] == 1) and (data['tau_1_matched_decay_mode'] == 0) and \
                (data['tau_0_matched_n_charged'] == 1) and (data['tau_1_matched_n_charged'] == 1) and \
                (data['tau_0_matched_n_neutral'] == 1) and (data['tau_1_matched_n_neutral'] == 0):
            return data['ditau_matched_CP_phi_star_cp_rho_ip']
        elif (data['tau_0_decay_mode'] == 2) and (data['tau_1_decay_mode'] == 0) and \
                (data['tau_0_matched_decay_mode'] == 2) and (data['tau_1_matched_decay_mode'] == 0) and \
                (data['tau_0_matched_n_charged'] == 1) and (data['tau_1_matched_n_charged'] == 1) and \
                (data['tau_1_matched_n_neutral'] == 0):
            return data['ditau_matched_CP_phi_star_cp_rho_ip']
        elif (data['tau_0_decay_mode'] == 0) and (data['tau_1_decay_mode'] == 2) and \
                (data['tau_0_matched_decay_mode'] == 0) and (data['tau_1_matched_decay_mode'] == 2) and \
                (data['tau_0_matched_n_charged'] == 1) and (data['tau_1_matched_n_charged'] == 1) and \
                (data['tau_0_matched_n_neutral'] == 0):
            return data['ditau_matched_CP_phi_star_cp_ip_rho']
        else:
            return 999 # arbitrary number 999 is assigned to the events where decay modes and matched decay modes are different

    df['phi_star'] = df.apply(Conditions, axis=1) # phi star is chosen according to the Conditions
    df = df.drop(['ditau_CP_phi_star_cp_a1_rho',
                  'ditau_CP_phi_star_cp_ip_ip',
                  'ditau_CP_phi_star_cp_ip_rho',
                  'ditau_CP_phi_star_cp_ip_rho_opt',
                  'ditau_CP_phi_star_cp_rho_ip',
                  'ditau_CP_phi_star_cp_rho_rho'], axis=1)
    df = df[df['phi_star'] < 999]
    df = df[df['phi_star'] > 0] # invalid events dropped: 999 = events with unmatched decay modes, 0 = events recorded as invalid at the data acquisition stage
    df.reset_index(drop=True, inplace=True)

    def Vector4(data):

        vectors = ['tau_0_matched_vis_charged_p4',
                   'tau_0_matched_vis_neutral_p4',
                   'tau_1_matched_vis_charged_p4',
                   'tau_1_matched_vis_neutral_p4']

        data = data[vectors]

        comps = ['X', 'Y', 'Z']
        for column in data.columns:
            for comp in comps:
                data[f'{column}{comp}'] = data[column].apply(lambda x: x['fP'][f'f{comp}'])
            data[f'{column}E'] = data[column].apply(lambda x: x['fE'])

        data = data.drop(vectors, axis=1)
        return data

    df_vec = Vector4(df)
    result = pd.concat([df, df_vec], axis=1)
    result = result.drop(['tau_0_matched_vis_charged_p4',
                          'tau_0_matched_vis_neutral_p4',
                          'tau_1_matched_vis_charged_p4',
                          'tau_1_matched_vis_neutral_p4'], axis=1)

    return result


def Events2(data_file): # different version of 'Events': a different set of branches chosen, removed the requirement for recorded tracks in the 'Conditions' function
    print("Reading file:", data_file)

    file = uproot.open(data_file)
    tree = file['NOMINAL']

    branches = ['ditau_matched_vis_mass',
                'tau_0_decay_mode',
                'tau_1_decay_mode',
                'tau_0_matched_decay_mode',
                'tau_1_matched_decay_mode',
                'NOMINAL_pileup_random_run_number',
                'event_is_bad_batman',
                'tau_0_matched_pdgId',
                'tau_1_matched_pdgId',
                'tau_0_jet_rnn_medium',
                'tau_1_jet_rnn_medium',
                'ditau_dr',
                'ditau_coll_approx_x0',
                'ditau_coll_approx_x1',
                'tau_0_q',
                'tau_1_q',
                'ditau_qxq',
                'tau_0_n_charged_tracks',
                'tau_1_n_charged_tracks',
                'ditau_deta',
                'ditau_CP_phi_star_cp_a1_rho',
                'ditau_CP_phi_star_cp_ip_ip',
                'ditau_CP_phi_star_cp_ip_rho',
                'ditau_CP_phi_star_cp_ip_rho_opt',
                'ditau_CP_phi_star_cp_rho_ip',
                'ditau_CP_phi_star_cp_rho_rho',
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
                'tau_0_p4',
                'tau_0_decay_neutral_p4',
                'tau_0_decay_charged_p4',
                'tau_1_p4',
                'tau_1_decay_neutral_p4',
                'tau_1_decay_charged_p4']

    df = tree.arrays(branches, library='pd')

    def Conditions(data):
        if (data['tau_0_decay_mode'] == 0) and (data['tau_1_decay_mode'] == 0) and \
                (data['tau_0_matched_decay_mode'] == 0) and (data['tau_1_matched_decay_mode'] == 0):
            return data['ditau_CP_phi_star_cp_ip_ip']
        elif (data['tau_0_decay_mode'] == 1) and (data['tau_1_decay_mode'] == 1) and \
                (data['tau_0_matched_decay_mode'] == 1) and (data['tau_1_matched_decay_mode'] == 1):
            return data['ditau_CP_phi_star_cp_rho_rho']
        elif (data['tau_0_decay_mode'] == 3) and (data['tau_1_decay_mode'] == 1) and \
                (data['tau_0_matched_decay_mode'] == 3) and (data['tau_1_matched_decay_mode'] == 1):
            return data['ditau_CP_phi_star_cp_a1_rho']
        elif (data['tau_0_decay_mode'] == 1) and (data['tau_1_decay_mode'] == 3) and \
                (data['tau_0_matched_decay_mode'] == 1) and (data['tau_1_matched_decay_mode'] == 3):
            return data['ditau_CP_phi_star_cp_a1_rho']
        elif (data['tau_0_decay_mode'] == 2) and (data['tau_1_decay_mode'] == 1) and \
                (data['tau_0_matched_decay_mode'] == 2) and (data['tau_1_matched_decay_mode'] == 1):
            return data['ditau_CP_phi_star_cp_rho_rho']
        elif (data['tau_0_decay_mode'] == 1) and (data['tau_1_decay_mode'] == 2) and \
                (data['tau_0_matched_decay_mode'] == 1) and (data['tau_1_matched_decay_mode'] == 2):
            return data['ditau_CP_phi_star_cp_rho_rho']
        elif (data['tau_0_decay_mode'] == 0) and (data['tau_1_decay_mode'] == 1) and \
                (data['tau_0_matched_decay_mode'] == 0) and (data['tau_1_matched_decay_mode'] == 1):
            return data['ditau_CP_phi_star_cp_ip_rho']
        elif (data['tau_0_decay_mode'] == 1) and (data['tau_1_decay_mode'] == 0) and \
                (data['tau_0_matched_decay_mode'] == 1) and (data['tau_1_matched_decay_mode'] == 0):
            return data['ditau_CP_phi_star_cp_rho_ip']
        elif (data['tau_0_decay_mode'] == 2) and (data['tau_1_decay_mode'] == 0) and \
                (data['tau_0_matched_decay_mode'] == 2) and (data['tau_1_matched_decay_mode'] == 0):
            return data['ditau_CP_phi_star_cp_rho_ip']
        elif (data['tau_0_decay_mode'] == 0) and (data['tau_1_decay_mode'] == 2) and \
                (data['tau_0_matched_decay_mode'] == 0) and (data['tau_1_matched_decay_mode'] == 2):
            return data['ditau_CP_phi_star_cp_ip_rho']
        else:
            return 999

    df['phi_star'] = df.apply(Conditions, axis=1)
    # df = df.drop(['ditau_CP_phi_star_cp_a1_rho',
    #               'ditau_CP_phi_star_cp_ip_ip',
    #               'ditau_CP_phi_star_cp_ip_rho',
    #               'ditau_CP_phi_star_cp_ip_rho_opt',
    #               'ditau_CP_phi_star_cp_rho_ip',
    #               'ditau_CP_phi_star_cp_rho_rho'], axis=1)
    df = df[df['phi_star'] < 999]
    df = df[df['phi_star'] > 0]
    df.reset_index(drop=True, inplace=True)

    def Vector4(data):

        vectors = ['tau_0_p4',
                   'tau_0_decay_neutral_p4',
                   'tau_1_p4',
                   'tau_1_decay_neutral_p4',
                   'tau_0_decay_charged_p4',
                   'tau_1_decay_charged_p4']

        data = data[vectors].copy()

        comps = ['X', 'Y', 'Z']
        for column in data.columns:
            for comp in comps:
                data.loc[:, f'{column}{comp}'] = data[column].apply(lambda x: x['fP'][f'f{comp}'])
            data.loc[:, f'{column}E'] = data[column].apply(lambda x: x['fE'])
            data.loc[:, f'{column}T'] = data[column].apply(lambda x: np.sqrt(x['fP']['fX']**2+x['fP']['fY']**2))

        data = data.drop(vectors, axis=1)
        return data

    df_vec = Vector4(df)
    result = pd.concat([df, df_vec], axis=1)
    result = result.drop(['tau_0_p4',
                          'tau_0_decay_neutral_p4',
                          'tau_1_p4',
                          'tau_1_decay_neutral_p4',
                          'tau_0_decay_charged_p4',
                          'tau_1_decay_charged_p4'], axis=1)

    return result

