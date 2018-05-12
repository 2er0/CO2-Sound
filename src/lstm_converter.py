import os
import numpy as np
import pandas as pd
import utils as ut

# config variables
mainPath = '/../data/'
wavPath = os.getcwd() + mainPath + 'wav/Train/'
npPath = os.getcwd() + mainPath + 'wav/Bin/'
train_path_short = os.getcwd() + mainPath + 'train_short.csv'
train_path_long = os.getcwd() + mainPath + 'train_long.csv'
test_path = os.getcwd() + mainPath + 'test.csv'

store_shape = (0, 41, 20)
label_shape = (0, 10)

# read csv mappings
train = pd.read_csv(train_path_long)
test = pd.read_csv(test_path)

# transform test data
test_waves_compact = ut.extract_by_ids_lstm(test['ID'].values, wavPath)
test_labels_short = ut.one_hot_encode_list(test['Class'].values)
test_waves = np.ndarray(store_shape)
test_labels = np.ndarray(label_shape)
for v, l in zip(test_waves_compact, test_labels_short):
    if len(v) < 1:
        continue
    test_waves = np.append(test_waves, v, axis=0)
    test_labels = np.append(test_labels, [l for _ in range(len(v))], axis=0)

# full_test_data.shape => (x, 41, 20)
full_test_data = np.array(list(zip(test_waves, test_labels)))
np.save(npPath + 'lstmTestData.npy', full_test_data)

# transform train data
train_waves_compact = ut.extract_by_ids_lstm(train['ID'].values, wavPath)
train_labels_short = ut.one_hot_encode_list(train['Class'].values)
train_waves = np.ndarray(store_shape)
train_labels = np.ndarray(label_shape)
for v, l in zip(train_waves_compact, train_labels_short):
    if len(v) < 1:
        continue
    train_waves = np.append(train_waves, v, axis=0)
    train_labels = np.append(train_labels, [l for _ in range(len(v))], axis=0)

full_train_data = np.array(list(zip(train_waves, train_labels)))
np.save(npPath + 'lstmTrainData.npy', full_train_data)
