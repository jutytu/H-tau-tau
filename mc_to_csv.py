# Creates csv files from the root files data using the 'Events2' function from prepare.py, saves them in folders named based on the files names.


from prepare import *


def mkdir_p(mypath): # for creating directories where the new files will be stores
    from errno import EEXIST
    from os import makedirs, path
    try:
        makedirs(mypath)
    except OSError as exc:  
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise

def mc_to_csv(plik):
    df = Events2(plik)
    mkdir_p(plik.strip(".txt.root"))
    df.to_csv(f'{plik.strip(".txt.root")}/{plik.strip(".txt.root")}.csv')
    print('Saved: ', f'{plik.strip(".txt.root")}/{plik.strip(".txt.root")}.csv')


pliki = {'ggH_CPeven.txt.root',
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
         'VBFH_unpol_d.txt.root',
         'ggH_CPeven_e.txt.root',
         'ggH_CPodd_e.txt.root',
         'ggH_unpol_e.txt.root',
         'VBFH_CPeven_e.txt.root',
         'VBFH_CPodd_e.txt.root',
         'VBFH_unpol_e.txt.root'} # the list of root files used for this analysis, for which csv files are created

for plik in pliki:
    mc_to_csv(plik)
