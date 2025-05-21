from myutil import DataLoader
from ACMCG.ACMCG import ACMCG
from joblib import Parallel, delayed
import os
import pandas as pd
import warnings

warnings.simplefilter('ignore')

def write_ari21(data, title, dir='ari21'):
    os.makedirs(dir, exist_ok=True)
    ari21 = {'dataset': [], 'ari': [], 'interaction': []}
    ari21['dataset'].append(title)
    ari21['ari'].append(data['ari'][-1])
    ari21['interaction'].append(data['interaction'][-1])
    df = pd.DataFrame(ari21)
    df.to_csv(f'{dir}/{title}.csv')


def add_ari21(data, title, ari21):
    ari21['dataset'].append(title)
    ari21['ari'].append(data['ari'][-1])
    ari21['interaction'].append(data['interaction'][-1])
    return ari21


def run(file: str) -> None:
    print(file)
    title = file.split('.')[0]
    data, label, k = DataLoader.get_data_from_local(dataDir + '/' + file)
    data = data.values
    datalen = len(data)
    print(f'{os.getpid()}\t run on {file}')
    print(f'run on ACMCG')
    ACMCG_ARI = ACMCG(data, label, title=title, q=datalen, k=k)
    ACMCGPath = 'result/ACMCG'
    os.makedirs(ACMCGPath, exist_ok=True)
    pd.DataFrame(ACMCG_ARI).to_csv(f'{ACMCGPath}/{title}.csv', index=False)


if __name__ == "__main__":
    dataDir = './data'
    files = os.listdir(dataDir)
    files = ['tae.csv']
    Parallel(n_jobs=-1, batch_size='auto')(delayed(run)(file) for file in files)


