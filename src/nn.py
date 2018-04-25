import os
import pandas as pd
import utils as ut

# global variables
validationSample = 20

# config variables
mainPath = '/../data/'
wavPath = os.getcwd() + mainPath + 'wav/Train/'
train_path_short = os.getcwd() + mainPath + 'train_short.csv'
train_path_long = os.getcwd() + mainPath + 'train_long.csv'
test_path = os.getcwd() + mainPath + 'test.csv'

# read train csv mapping
train = pd.read_csv(train_path_short)

# split data
validate = train.sample(validationSample)
train = train.drop(validate.index)

# refactor index on data-frames
train = train.reset_index(drop=True)
validate = validate.reset_index(drop=True)

# read wav's
validate_wavs = ut.load_by_ids(validate['ID'].values, wavPath)

# test plot of converted wav's
#ut.plot_waves(validate['Class'].values, validate_wavs)
#ut.plot_spectrogram(validate['Class'].values, validate_wavs)
#ut.plot_log_power_spectrogram(validate['Class'].values, validate_wavs)

# all featurs in a list(1) of list(2)
# list(2) contains all feature values from the transformed signal in an np.hstack
# one list(2) has currently a size of 193
res = ut.extract_by_ids(validate['ID'].values, wavPath)

print(ut.one_hot_encode(validate.iloc[0]['Class']))

