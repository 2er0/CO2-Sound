import os
import numpy as np
import pandas as pd
import utils as ut

# # config variables
mainPath = '/../data/'
wavPath = os.getcwd() + mainPath + 'wav/Train/'
npPath = os.getcwd() + mainPath + 'wav/Bin/'
train_path_short = os.getcwd() + mainPath + 'train_short.csv'
train_path_long = os.getcwd() + mainPath + 'train_long.csv'
test_path = os.getcwd() + mainPath + 'test.csv'

# # read train csv mapping
train = pd.read_csv(train_path_long)
test = pd.read_csv(test_path)

test_waves = ut.extract_by_ids(test['ID'].values, wavPath)
test_waves = ut.feature_normalize(test_waves)
test_labels = ut.one_hot_encode_list(test['Class'].values)

full_test_data = np.array(list(zip(test_waves, test_labels)))
np.save(npPath + 'nnTestData.npy', full_test_data)

train_waves = ut.extract_by_ids(train['ID'].values, wavPath)
train_waves = ut.feature_normalize(train_waves)
train_labels = ut.one_hot_encode_list(train['Class'].values)

full_train_data = np.array(list(zip(train_waves, train_labels)))
np.save(npPath + 'nnTrainData.npy', full_train_data)
