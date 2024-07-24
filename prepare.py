# import ROOT as root
import numpy as np
import uproot
import pandas as pd
import math
# import awkward


def AHHH1(data_file):
    print("Reading file:", data_file)

    file = uproot.open(data_file)
    tree = file['NOMINAL']

    branches = [
        'ditau_matched_vis_mass',
        'ditau_met_min_dphi',
        'ditau_dr',
        'ditau_deta',
        'ditau_dphi',
        'ditau_mmc_mlm_m',
        'ditau_higgspt',
        'n_vx',
        'n_actual_int',
        'n_avg_int'
    ]

    df = tree.arrays(branches, library='pd')
    return df


def AHHH2(data_file):
    print("Reading file:", data_file)

    file = uproot.open(data_file)
    tree = file['NOMINAL']

    vectors = ['tau_0_p4',
               'tau_1_p4',
               'met_p4']

    df = tree.arrays(vectors, library='pd')

    comps = ['X', 'Y', 'Z']
    for column in df.columns:
        for comp in comps:
            df[f'{column}{comp}'] = df[column].apply(lambda x: x['fP'][f'f{comp}'])
        df[f'{column}E'] = df[column].apply(lambda x: x['fE'])

    df['tau_0_eta'] = df.apply(lambda row: math.atanh(row['tau_0_p4Z'] / row['tau_0_p4E']), axis=1)
    df['tau_1_eta'] = df.apply(lambda row: math.atanh(row['tau_1_p4Z'] / row['tau_1_p4E']), axis=1)

    df = df.drop(vectors, axis=1)
    return df


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
            return 999

    df['phi_star'] = df.apply(Conditions, axis=1)
    df = df.drop(['ditau_CP_phi_star_cp_a1_rho',
                  'ditau_CP_phi_star_cp_ip_ip',
                  'ditau_CP_phi_star_cp_ip_rho',
                  'ditau_CP_phi_star_cp_ip_rho_opt',
                  'ditau_CP_phi_star_cp_rho_ip',
                  'ditau_CP_phi_star_cp_rho_rho'], axis=1)
    df = df[df['phi_star'] < 999]
    df = df[df['phi_star'] > 0]
    df.reset_index(drop=True, inplace=True)

    def vector4(data):

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

    df_vec = vector4(df)
    result = pd.concat([df, df_vec], axis=1)
    result = result.drop(['tau_0_matched_vis_charged_p4',
                          'tau_0_matched_vis_neutral_p4',
                          'tau_1_matched_vis_charged_p4',
                          'tau_1_matched_vis_neutral_p4'], axis=1)

    return result


def Events2(data_file):
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

    def vector4(data):

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

    df_vec = vector4(df)
    result = pd.concat([df, df_vec], axis=1)
    result = result.drop(['tau_0_p4',
                          'tau_0_decay_neutral_p4',
                          'tau_1_p4',
                          'tau_1_decay_neutral_p4',
                          'tau_0_decay_charged_p4',
                          'tau_1_decay_charged_p4'], axis=1)

    return result


def Conditions2(data):
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
