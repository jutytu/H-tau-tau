# Plotting some more quantities. This function creates folders with multiple plots and confusion matrices for different data files, listed as 'files' under the
# function definition. Some of the folders (size limit) can be found in this repository.


import seaborn as sns
import matplotlib.pyplot as plt
from prepare import *
import matplotlib.lines as mlines
import math


def Yrs(plik):
    pd.set_option('mode.chained_assignment', None)
    df = Events2(plik)
    pd.set_option('display.max_columns', None)


    # number of events
    print(f'num of events {plik}: ', len(df))
    num = len(df)

    # preselection
    df = df[(df['NOMINAL_pileup_random_run_number'] > 0) & (df['event_is_bad_batman'] == 0) &
            (((abs(df['tau_0_matched_pdgId']) < 7) | (df['tau_0_matched_pdgId'] == 21)) == 0) &
            (((abs(df['tau_1_matched_pdgId']) < 7) | (df['tau_1_matched_pdgId'] == 21)) == 0) &
            (df['tau_0_jet_rnn_medium'] == 1) & (df['tau_1_jet_rnn_medium'] == 1) & (df['ditau_dr'] > 0.6) &
            (df['ditau_coll_approx_x0'] > 0.1) & (df['ditau_coll_approx_x0'] < 1.4) &
            (df['ditau_coll_approx_x1'] > 0.1) & (df['ditau_coll_approx_x1'] < 1.4) &
            (abs(df['tau_0_q']) == 1) & (abs(df['tau_1_q']) == 1) & (df['ditau_qxq'] == -1) &
            ((df['tau_0_n_charged_tracks'] == 1) | (df['tau_0_n_charged_tracks'] == 3)) &
            ((df['tau_1_n_charged_tracks'] == 1) | (df['tau_1_n_charged_tracks'] == 3)) &
            (df['ditau_deta'] < 1.5)]

    print(f'num of events {plik} after preselection: ', len(df))
    print(f'acceptance {plik}: ', len(df)/num, '\n')


    # decay mode distributions

    def mkdir_p(mypath):
        from errno import EEXIST
        from os import makedirs,path
        try:
            makedirs(mypath)
        except OSError as exc: 
            if exc.errno == EEXIST and path.isdir(mypath):
                pass
            else: raise

    mkdir_p(plik.strip(".txt.root"))
    plt.figure(figsize=(5, 5))
    counts = df['tau_0_decay_mode'].value_counts()
    plt.scatter(counts.index, counts, s=4500, color='red', alpha=0.7, marker=1)
    plt.errorbar(x=counts.index + 0.5, y=counts, yerr=math.sqrt(len(df)), fmt='none', ecolor='black', capsize=5)
    plt.xlabel('tau_0_decay_mode')
    plt.ylabel('counts')
    blue_line = mlines.Line2D([], [], color='red', markersize=15, label=plik.strip('.txt.root'))
    plt.legend(handles=[blue_line])
    plt.xticks([0, 1, 2, 3, 4])
    plt.savefig(f'{plik.strip(".txt.root")}/{plik.strip(".txt.root")}_tau0_dm.png')
    plt.show()

    plt.figure(figsize=(5, 5))
    counts = df['tau_1_decay_mode'].value_counts()
    plt.scatter(counts.index, counts, s=4500, color='red', alpha=0.7, marker=1)
    plt.errorbar(x=counts.index + 0.5, y=counts, yerr=math.sqrt(len(df)), fmt='none', ecolor='black', capsize=5)
    plt.xlabel('tau_1_decay_mode')
    plt.ylabel('counts')
    blue_line = mlines.Line2D([], [], color='red', markersize=15, label=plik.strip('.txt.root'))
    plt.legend(handles=[blue_line])
    plt.xticks([0, 1, 2, 3, 4])
    plt.savefig(f'{plik.strip(".txt.root")}/{plik.strip(".txt.root")}_tau1_dm.png')
    plt.show()


    # confusion matrix of decay modes and matched decay modes

    df2 = df[['tau_0_decay_mode', 'tau_0_matched_decay_mode']]
    df2 = df2[df2['tau_0_matched_decay_mode']<4]

    pivot_table = df2.pivot_table(index='tau_0_decay_mode', columns='tau_0_matched_decay_mode', aggfunc=len, fill_value=0)

    plt.figure(figsize=(6, 5))
    heatmap = sns.heatmap(pivot_table, annot=True, cmap='Purples', fmt='g')
    heatmap.xaxis.tick_top()
    heatmap.xaxis.set_label_position('top')
    heatmap.set_xlabel('tau_0_matched_decay_mode')
    heatmap.set_ylabel('tau_0_decay_mode')

    plt.savefig(f'{plik.strip(".txt.root")}/{plik.strip(".txt.root")}_tau0_cm.png')
    plt.show()
    ###
    df2 = df[['tau_1_decay_mode', 'tau_1_matched_decay_mode']]
    df2 = df2[df2['tau_1_matched_decay_mode']<4]

    pivot_table = df2.pivot_table(index='tau_1_decay_mode', columns='tau_1_matched_decay_mode', aggfunc=len, fill_value=0)

    plt.figure(figsize=(6, 5))
    heatmap = sns.heatmap(pivot_table, annot=True, cmap='Purples', fmt='g')
    heatmap.xaxis.tick_top()
    heatmap.xaxis.set_label_position('top')
    heatmap.set_xlabel('tau_1_matched_decay_mode')
    heatmap.set_ylabel('tau_1_decay_mode')

    plt.savefig(f'{plik.strip(".txt.root")}/{plik.strip(".txt.root")}_tau1_cm.png')
    plt.show()


    # phi distributions

    dict = {'1p0n 1p0n': [0, 0],
            '1p1n 1p0n': [1, 0],
            '1p0n 1p1n': [0, 1],
            '1p1n 1p1n': [1, 1],
            '1p0n 1pXn': [0, 2],
            '1pXn 1p0n': [2, 0],
            '1p1n 1pXn': [1, 2],
            '1pXn 1p1n': [2, 1],
            '1p1n 3p0n': [1, 3],
            '3p0n 1p1n': [3, 1]}

    plt.figure(figsize=(5, 4))
    hist, bins = np.histogram(df['phi_star'], bins=20) 
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.hist(df['phi_star'], bins=20, edgecolor='black', alpha=0.7)

    scatter_x = bin_centers
    scatter_y = hist
    plt.clf()
    plt.scatter(scatter_x, scatter_y, s=320, color='red', marker=1)
    plt.xlabel('phi_star')
    plt.ylabel('counts')
    blue_line = mlines.Line2D([], [], color='red', markersize=15, label=plik.strip('.txt.root'))
    plt.legend(handles=[blue_line])
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7])

    plt.savefig(f'{plik.strip(".txt.root")}/{plik.strip(".txt.root")}_phi_star.png')
    plt.show()


    # kinematic variables

    plt.figure(figsize=(5, 5))
    hist, bins = np.histogram(df['ditau_matched_vis_mass'], bins=20) 
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.hist(df['ditau_matched_vis_mass'], bins=20, edgecolor='black', alpha=0.7)

    scatter_x = bin_centers
    scatter_y = hist
    plt.clf()
    plt.scatter(scatter_x, scatter_y, s=280, color='red', marker=1)
    plt.xlabel('ditau_matched_vis_mass')
    plt.ylabel('counts')
    blue_line = mlines.Line2D([], [], color='red', markersize=15, label=plik.strip('.txt.root'))
    plt.legend(handles=[blue_line])

    plt.savefig(f'{plik.strip(".txt.root")}/{plik.strip(".txt.root")}_vis_mass.png')
    plt.show()
    ###
    plt.figure(figsize=(5, 5))
    hist, bins = np.histogram(df[(df['tau_0_p4T']>0) & (df['tau_0_p4T']<250)]['tau_0_p4T'], bins=30)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.hist(df[(df['tau_0_p4T']>0) & (df['tau_0_p4T']<250)]['tau_0_p4T'], bins=30, edgecolor='black', alpha=0.7)

    scatter_x = bin_centers
    scatter_y = hist
    plt.clf()
    plt.scatter(scatter_x, scatter_y, s=50, color='red', marker=1)

    plt.xlabel('tau_0_pT')
    plt.ylabel('counts')
    blue_line = mlines.Line2D([], [], color='red', markersize=15, label=plik.strip('.txt.root'))
    plt.legend(handles=[blue_line])

    plt.savefig(f'{plik.strip(".txt.root")}/{plik.strip(".txt.root")}_tau_0_pT.png')
    plt.show()
    ###
    plt.figure(figsize=(5, 5))
    hist, bins = np.histogram(df[(df['tau_1_p4T'] > 0) & (df['tau_1_p4T'] < 250)]['tau_1_p4T'], bins=30)  
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.hist(df[(df['tau_1_p4T'] > 0) & (df['tau_1_p4T'] < 250)]['tau_1_p4T'], bins=30, edgecolor='black', alpha=0.7)

    scatter_x = bin_centers
    scatter_y = hist
    plt.clf()
    plt.scatter(scatter_x, scatter_y, s=50, color='red', marker=1)

    plt.xlabel('tau_1_pT')
    plt.ylabel('counts')
    blue_line = mlines.Line2D([], [], color='red', markersize=15, label=plik.strip('.txt.root'))
    plt.legend(handles=[blue_line])

    plt.savefig(f'{plik.strip(".txt.root")}/{plik.strip(".txt.root")}_tau_1_pT.png')
    plt.show()

files = {'ggH_CPeven_e.txt.root',
         'ggH_CPodd_e.txt.root',
         'ggH_unpol_e.txt.root',
         'VBFH_CPeven_e.txt.root',
         'VBFH_CPodd_e.txt.root',
         'VBFH_unpol_e.txt.root'}
         'ggH_CPeven.txt.root',
         'ggH_CPodd.txt.root',
         'ggH_unpol.txt.root',
         'VBFH_CPeven.txt.root',
         'VBFH_CPodd.txt.root',
         'VBFH_unpol.txt.root',
         'ggH_CPeven_d.txt.root',
         'ggH_CPodd_d.txt.root',
         'ggH_unpol_d.txt.root',
         'VBFH_CPeven_d.txt.root',
         'VBFH_CPodd_d.txt.root',
         'VBFH_unpol_d.txt.root'}

for i in files:
    Yrs(i)
