import os
import numpy as np
import pandas as pd
import utils as ut

# config variables
mainPath = '/../data/'
kPath = mainPath + 'wav/8k/UrbanSound8K/'
wavPath = os.getcwd() + kPath + 'audio/'
npPath = os.getcwd() + mainPath + 'wav/Bin/'
data_path = os.getcwd() + kPath + 'metadata/UrbanSound8K.csv'

store_shape = (0, 41, 20)
label_shape = (0, 10)

# read csv mappings
data = pd.read_csv(data_path)

# transform data
soundLabeled = ut.extract_by_ids_lstm_8k(data, wavPath)

data_waves = np.ndarray(store_shape)
data_labels = np.ndarray(label_shape)
for v, l in soundLabeled:
    if len(v) < 1:
        continue
    data_waves = np.append(data_waves, v, axis=0)
    data_labels = np.append(data_labels, [l for _ in range(len(v))], axis=0)

# full_test_data.shape => (x, 41, 20)
full_test_data = np.array(list(zip(data_waves, data_labels)))
np.save(npPath + 'lstm8kTrainData.npy', full_test_data)