import os
import pandas as pd
import utlis as ut

trainSample = 20

mainPath = '/../data/'
wavPath = os.getcwd() + mainPath + 'wav/Train/'
train_path_short = os.getcwd() + mainPath + 'train_short.csv'
train_path_long = os.getcwd() + mainPath + 'train_long.csv'
test_path = os.getcwd() + mainPath + 'test.csv'

train_short = pd.read_csv(train_path_short)

validate = pd.DataFrame(train_short.sample(5))
train_short = pd.DataFrame(train_short.drop(validate.index))

train_short = pd.DataFrame(train_short.reset_index())
validate = pd.DataFrame(validate.reset_index())

train_wav = ut.load_by_id(validate['ID'].values, wavPath)

ut.plot_waves(validate['Class'].values, train_wav)
ut.plot_specgram(validate['Class'].values, train_wav)
ut.plot_log_power_specgram(validate['Class'].values, train_wav)

#train_data = ut.load_sounds(train_short['ID'])

