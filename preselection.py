# This function puts preselection criteria on a chosen root file, calculates acceptance and shows phi star distributions after the preselection.

from prepare import *
import matplotlib.pyplot as plt

def Criteria(data):
        df = Events2(data)
        pd.set_option('display.max_columns', None)
        print(df.head())
        len0 = len(df) # initial number of events
        dict = {'1p0n 1p0n': [0, 0],
                '1p1n 1p0n': [1, 0],
                '1p0n 1p1n': [0, 1],
                '1p1n 1p1n': [1, 1],
                '1p0n 1pXn': [0, 2],
                '1pXn 1p0n': [2, 0],
                '1p1n 1pXn': [1, 2],
                '1pXn 1p1n': [2, 1],
                '1p1n 3p0n': [1, 3],
                '3p0n 1p1n': [3, 1]} # dictionary with all the possible combinations of decay modes of both tau leptons
        
        events = []
        for key, value in dict.items():
                n = len(df[(df['tau_0_decay_mode'] == value[0]) & (df['tau_1_decay_mode'] == value[1])])
                events.append(n)
                print(key, ' ', n) # calculates the number of events for each decay mode combination and stores them in a list
        
        print('number of events: ', len(df))
        
        df = df[(df['NOMINAL_pileup_random_run_number'] > 0) & (df['event_is_bad_batman'] == 0) &
                (((abs(df['tau_0_matched_pdgId']) < 7) | (df['tau_0_matched_pdgId'] == 21)) == 0) &
                (((abs(df['tau_1_matched_pdgId']) < 7) | (df['tau_1_matched_pdgId'] == 21)) == 0) &
                (df['tau_0_jet_rnn_medium'] == 1) & (df['tau_1_jet_rnn_medium'] == 1) & (df['ditau_dr'] > 0.6) &
                (df['ditau_coll_approx_x0'] > 0.1) & (df['ditau_coll_approx_x0'] < 1.4) &
                (df['ditau_coll_approx_x1'] > 0.1) & (df['ditau_coll_approx_x1'] < 1.4) &
                (abs(df['tau_0_q']) == 1) & (abs(df['tau_1_q']) == 1) & (df['ditau_qxq'] == -1) &
                ((df['tau_0_n_charged_tracks'] == 1) | (df['tau_0_n_charged_tracks'] == 3)) &
                ((df['tau_1_n_charged_tracks'] == 1) | (df['tau_0_n_charged_tracks'] == 3)) &
                (df['ditau_deta'] < 1.5)] # all the criteria used here come from the ATLAS documentation
        
        
        print('number of events after preselection: ', len(df))
        print('acceptance: ', len(df)/len0) # acceptance = events after preselection / events before preselection
        
        events2 = 0 # iterator
        for key, value in dict.items():
                n = len(df[(df['tau_0_decay_mode'] == value[0]) & (df['tau_1_decay_mode'] == value[1])])
                print('events after preselection for ', key, ': ', n)
                print('acceptance ', key, ': ', n/events[events2])
                events2 += 1 # prints out acceptances for all the decay mode combinations using the initial numbers of events stored in the 'events' list
        
        n = 1
        for key, value in dict.items():
                if key == '1p1n 3p0n':
                        continue
                plt.subplot(3, 3, n)
                plt.hist(df[(df['tau_0_decay_mode'] == value[0]) & (df['tau_1_decay_mode'] == value[1])]['phi_star'], bins=30,
                         weights=df[(df['tau_0_decay_mode'] == value[0]) & (df['tau_1_decay_mode'] == value[1])]['tauspinner_HCP_Theta_0'])
                plt.title(f'phi_star {key}')
                plt.xticks([])
                n += 1
        plt.show() # plots phi star distributions for different decay mode combinations (except 1p1n 3p0n - low quality data) using weights for mixing angle = 0 ('tauspinner_HCP_Theta_0')
        
  
