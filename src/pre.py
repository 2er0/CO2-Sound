import os
import numpy as np
import pandas as pd
import utils as ut


def load(wav_source: bool = False, validation_sample: int = 30, data_type: str = 'nn'):
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
        train = pd.DataFrame(np.load(npyPath + data_type + 'TrainData.npy'))
        test = pd.DataFrame(np.load(npyPath + data_type + 'TestData.npy'))

    # split data
    validate = train.sample(validation_sample)
    train = train.drop(validate.index)

    # refactor index on data-frames
    train = train.reset_index(drop=True)
    validate = validate.reset_index(drop=True)

    if wav_source:
        if data_type != 'nn':
            raise ValueError('not supported')
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

        train_waves = ut.extract_by_ids_full(train['ID'].values, wavPath)
        train_waves = ut.feature_normalize(train_waves)
        train_labels = ut.one_hot_encode_list(train['Class'].values)

        validate_waves = ut.extract_by_ids_full(validate['ID'].values, wavPath)
        validate_waves = ut.feature_normalize(validate_waves)
        validate_labels = ut.one_hot_encode_list(validate['Class'].values)

        test_waves = ut.extract_by_ids_full(test['ID'].values, wavPath)
        test_waves = ut.feature_normalize(test_waves)
        test_labels = ut.one_hot_encode_list(test['Class'].values)

    else:
        #if data_type == 'nn':
        test_waves = np.vstack(test[0])
        test_labels = np.vstack(test[1])

        train_waves = np.vstack(train[0])
        train_labels = np.vstack(train[1])

        validate_waves = np.vstack(validate[0])
        validate_labels = np.vstack(validate[1])
        """
        else:
            train_waves = np.ndarray(shape=(0, 60, 41, 2))
            train_labels = np.ndarray(shape=(0, 10))

            for i in range(len(train)):
                subs = len(train[0][i])
                if subs < 1:
                    continue
                train_waves = np.append(train_waves, train[0][i], axis=0)
                train_labels = np.append(train_labels, [train[1][i] for _ in range(subs)], axis=0)

            if not train_waves.shape[0] == train_labels.shape[0]:
                raise ValueError('Train data have not the same shape')

            validate_waves = np.ndarray(shape=(0, 60, 41, 2))
            validate_labels = np.ndarray(shape=(0, 10))

            for i in range(len(validate)):
                subs = len(validate[0][i])
                if subs < 1:
                    continue
                validate_waves = np.append(validate_waves, validate[0][i], axis=0)
                validate_labels = np.append(validate_labels, [validate[1][i] for _ in range(subs)], axis=0)

            if not validate_waves.shape[0] == validate_labels.shape[0]:
                raise ValueError('Validate data have not the same shape')

            test_waves = np.ndarray(shape=(0, 60, 41, 2))
            test_labels = np.ndarray(shape=(0, 10))

            for i in range(len(test)):
                subs = len(test[0][i])
                if subs < 1:
                    continue
                test_waves = np.append(test_waves, test[0][i], axis=0)
                test_labels = np.append(test_labels, [test[1][i] for _ in range(subs)], axis=0)

            if not test_waves.shape[0] == test_labels.shape[0]:
                raise ValueError('Test data have not the same shape')
        """

    return train_waves, train_labels, \
           validate_waves, validate_labels, \
           test_waves, test_labels
