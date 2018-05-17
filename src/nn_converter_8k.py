import os
import numpy as np
import pandas as pd
import utils as ut

# # config variables
mainPath = '/../data/'
kPath = mainPath + 'wav/8k/UrbanSound8K/'
wavPath = os.getcwd() + kPath + 'audio/'
npPath = os.getcwd() + mainPath + 'wav/Bin/'
data_path = os.getcwd() + kPath + 'metadata/UrbanSound8K.csv'

# read csv mappings
data = pd.read_csv(data_path)

# transform data
soundLabeled = ut.extract_by_ids_full_8k(data, wavPath)

np.save(npPath + 'nn8kTrainData.npy', soundLabeled)
