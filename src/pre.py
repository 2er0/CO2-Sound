import os
import numpy as np
import pandas as pd
import utils as ut


def load(shape_type, data_type: str = 'nn'):
    # # config variables
    mainPath = '/../data/'
    wavPath = os.getcwd() + mainPath + 'wav/Train/'
    npyPath = os.getcwd() + mainPath + 'wav/Bin/'

    train = np.load(npyPath + data_type + 'TrainData.npy')
    test = np.load(npyPath + data_type + 'TestData.npy')

    # split data
    # validate = train.sample(validation_sample)
    # train = train.drop(validate.index)

    # refactor index on data-frames
    # train = train.reset_index(drop=True)
    # validate = validate.reset_index(drop=True)

    # if data_type == 'nn':
    # test_waves = np.vstack(test[0])
    # test_labels = np.vstack(test[1])

    if data_type == 'nn':
        test_waves = np.vstack(test[:, 0])
        test_labels = np.vstack(test[:, 1])
    else:
        test = pd.DataFrame(test)
        test_waves = np.vstack([row[0][np.newaxis, :] for _, row in test.iterrows()])
        test_labels = np.vstack(test[1])

    if not test_waves[0].shape == shape_type:
        raise ValueError("dim does not match {} != {}".
                         format(test_waves[0].shape, shape_type))

    if data_type == 'nn':
        train_waves = np.vstack(train[:, 0])
        train_labels = np.vstack(train[:, 1])
    else:
        train = pd.DataFrame(train)
        train_waves = np.vstack([row[0][np.newaxis, :] for _, row in train.iterrows()])
        train_labels = np.vstack(train[1])

    if not train_waves[0].shape == shape_type:
        raise ValueError("dim does not match {} != {}".
                         format(train_waves[0].shape, shape_type))

    """
    # validate_waves = np.vstack(validate[0])
    # validate_labels = np.vstack(validate[1])

    validate_waves = np.array(validate[0])
    validate_labels = np.array(validate[1])

    if not validate_waves[0].shape == shape_type:
        raise ValueError("dim does not match {} != {}".
                         format(validate_waves[0].shape, shape_type))
    """

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
           test_waves, test_labels #\
            #validate_waves, validate_labels, \
