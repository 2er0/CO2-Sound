import os
import numpy as np
import pandas as pd
import utils as ut

# # config variables
mainPath = '/../data/'
wavPath = os.getcwd() + mainPath + 'wav/Train/'
npyPath = os.getcwd() + mainPath + 'wav/Bin/'


def load(shape_type, data_type: str = 'nn'):
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

    return train_waves, train_labels, \
           test_waves, test_labels


def load8k(shape_type, data_type: str = 'nn8k'):
    train = pd.DataFrame(np.load(npyPath + data_type + 'TrainData.npy'))

    train = train.sample(frac=1).reset_index(drop=True)

    testCount = int(train.shape[0] / 5)
    test = train.sample(testCount, replace=True)
    train = train.drop(test.index)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    if data_type == 'nn8k':
        train_waves = np.vstack(train[0])
        train_labels = np.vstack(train[1])

        test_waves = np.vstack(test[0])
        test_labels = np.vstack(test[1])
    else:
        train_waves = np.vstack([row[0][np.newaxis, :] for _, row in train.iterrows()])
        train_labels = np.vstack(train[1])

        test_waves = np.vstack([row[0][np.newaxis, :] for _, row in test.iterrows()])
        test_labels = np.vstack(test[1])

    if not train_waves.shape[1:] == shape_type \
            or not test_waves.shape[1:] == shape_type:
        raise ValueError("Dims does not match")

    return train_waves, train_labels, \
           test_waves, test_labels