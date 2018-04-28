import os
import numpy as np
import pandas as pd
import utils as ut


def load(wav_source: bool = False, validation_sample: int = 30):
    # # config variables
    mainPath = '/../data/'
    wavPath = os.getcwd() + mainPath + 'wav/Train/'
    npyPath = os.getcwd() + mainPath + 'wav/Bin/'

    if wav_source:
        train_path_short = os.getcwd() + mainPath + 'train_short.csv'
        train_path_long = os.getcwd() + mainPath + 'train_long.csv'
        test_path = os.getcwd() + mainPath + 'test.csv'

        # # read train csv mapping
        train = pd.read_csv(train_path_short)
        test = pd.read_csv(test_path)

    else:
        # contains feature values from fft and one_hot encoded classes
        train = pd.DataFrame(np.load(npyPath + 'trainData.npy'))
        test = pd.DataFrame(np.load(npyPath + 'testData.npy'))

    # split data
    validate = train.sample(validation_sample)
    train = train.drop(validate.index)

    # refactor index on data-frames
    train = train.reset_index(drop=True)
    validate = validate.reset_index(drop=True)

    if wav_source:
        # read wav's
        # validate_wavs = ut.load_by_ids(validate['ID'].values, wavPath)

        # test plot of converted wav's
        # ut.plot_waves(validate['Class'].values, validate_wavs)
        # ut.plot_spectrogram(validate['Class'].values, validate_wavs)
        # ut.plot_log_power_spectrogram(validate['Class'].values, validate_wavs)

        # all featurs in a list(1) of list(2)
        # list(2) contains all feature values from the transformed signal in an np.hstack
        # one list(2) has currently a size of 193
        # after normalize np.ndarray((x,193))

        train_waves = ut.extract_by_ids(train['ID'].values, wavPath)
        train_waves = ut.feature_normalize(train_waves)
        train_labels = ut.one_hot_encode_list(train['Class'].values)

        validate_waves = ut.extract_by_ids(validate['ID'].values, wavPath)
        validate_waves = ut.feature_normalize(validate_waves)
        validate_labels = ut.one_hot_encode_list(validate['Class'].values)

        test_waves = ut.extract_by_ids(test['ID'].values, wavPath)
        test_waves = ut.feature_normalize(test_waves)
        test_labels = ut.one_hot_encode_list(test['Class'].values)
    else:
        train_waves = np.vstack(train[0])
        train_labels = np.vstack(train[1])

        validate_waves = np.vstack(validate[0])
        validate_labels = np.vstack(validate[1])

        test_waves = np.vstack(test[0])
        test_labels = np.vstack(test[1])

    return train_waves, train_labels, \
           validate_waves, validate_labels, \
           test_waves, test_labels
