from ucimlrepo import fetch_ucirepo, list_available_datasets
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
import numpy as np
import pandas as pd


def preprocess_data(dataset_path: str, doPerturb: bool):
    df = pd.read_csv(dataset_path)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    data = df.iloc[:, :-1].values
    data = data.astype(float)
    label = df.iloc[:, -1]
    le = LabelEncoder()
    le = le.fit(label)
    label = le.transform(label)

    k = le.classes_.shape[0]

    if doPerturb:
        random_matrix = np.random.rand(data.shape[0], data.shape[1]) * 1e-7
        data = data + random_matrix
    data = pd.DataFrame(data)
    return data, label, k


def get_data_from_local(dataset_path: str, doPerturb: bool = True):
    return preprocess_data(dataset_path, doPerturb)


def list_available_datasets():
    datasets = list_available_datasets()
    print(datasets)


def load_data(name: str):
    original = fetchData(name)
    X, y, n_classes = process_data(original)
    return X, y, n_classes


def process_data(original: pd.DataFrame):
    print('begain process_data', original.shape)

    original.dropna(inplace=True)
    original.drop_duplicates(inplace=True)
    print('after drop_duplicates', original.shape)
    X = original.iloc[:, :-1].values
    y = original.iloc[:, -1].values
    X = MinMaxScaler().fit_transform(X)
    y = LabelEncoder().fit_transform(y)
    yu = np.unique(y)
    n_classes = yu.shape[0]
    return X, y, n_classes


def fetchData(name: str):
    heart_disease = fetch_ucirepo(name=name)
    print(heart_disease.metadata.abstract)
    print(heart_disease.metadata.additional_info.summary)

    return heart_disease.data.original

def adjConcat(a, b):
    lena = len(a)
    lenb = len(b)
    left = np.row_stack((a, np.zeros((lenb, lena))))
    right = np.row_stack((np.zeros((lena, lenb)), b))
    result = np.hstack((left, right)) 
    return result


def get_syc_data(blockSize: list, blockNum: int):
    if len(blockSize) < 1:
        return None
    result = np.ones((blockSize[0], blockSize[0]))
    for _ in range(blockNum):
        a = np.ones((blockSize[0], blockSize[0]))
        result = adjConcat(result, a)
    return result, _, blockNum


if __name__ == '__main__':
    raise Exception('This is a module, not a script!')
    data = ['Balance', 'banknote', 'breast', 'dermatology', 'diabetes', 'ecoli', 'glass', 'haberman', 'ionosphere', 'iris',
            'led', 'mfeat karhunen', 'mfeat zernike', 'musk', 'pima', 'seeds', 'segment', 'soybean', 'thyroid', 'vehicle', 'wine']
    for i in data:
        print(i)
        X, y, n_classes = load_data('i')
        print(np.hstack((X, y.reshape(-1, 1))).shape)
