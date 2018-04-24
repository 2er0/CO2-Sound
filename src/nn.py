import os
import pandas as pd
import utlis as ut

# global variables
validationSample = 20

# config variables
mainPath = '/../data/'
wavPath = os.getcwd() + mainPath + 'wav/Train/'
train_path_short = os.getcwd() + mainPath + 'train_short.csv'
train_path_long = os.getcwd() + mainPath + 'train_long.csv'
test_path = os.getcwd() + mainPath + 'test.csv'

# read train csv mapping
train_short = pd.read_csv(train_path_short)

# split data
validate = pd.DataFrame(train_short.sample(validationSample))
train_short = pd.DataFrame(train_short.drop(validate.index))

# refactor index on data-frames
train_short = pd.DataFrame(train_short.reset_index())
validate = pd.DataFrame(validate.reset_index())

# read wav's
validate_wavs = ut.load_by_id(validate['ID'].values, wavPath)

# test plot of converted wav's
#ut.plot_waves(validate['Class'].values, validate_wavs)
#ut.plot_spectrogram(validate['Class'].values, validate_wavs)
#ut.plot_log_power_spectrogram(validate['Class'].values, validate_wavs)

#train_data = ut.load_sounds(train_short['ID'])
