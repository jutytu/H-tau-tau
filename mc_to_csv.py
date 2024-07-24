# przerabia te glowne pliki root na csv (najpierw df - Events2 z prepare) i wklada do folderow

from prepare import *
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def mkdir_p(mypath):
    from errno import EEXIST
    from os import makedirs, path
    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise

def mc_to_csv(plik):
    df = Events2(plik)
    mkdir_p(plik.strip(".txt.root"))
    df.to_csv(f'{plik.strip(".txt.root")}/{plik.strip(".txt.root")}.csv')
    print('Saved: ', f'{plik.strip(".txt.root")}/{plik.strip(".txt.root")}.csv')

# mc_to_csv('VBFH_unpol_d.txt.root')

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
         'VBFH_unpol_e.txt.root'}

for plik in pliki:
    mc_to_csv(plik)
