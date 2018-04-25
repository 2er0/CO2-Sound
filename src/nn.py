import os
import pandas as pd
import utils as ut

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# # global variables
validationSample = 30

# # config variables
mainPath = '/../data/'
wavPath = os.getcwd() + mainPath + 'wav/Train/'
train_path_short = os.getcwd() + mainPath + 'train_short.csv'
train_path_long = os.getcwd() + mainPath + 'train_long.csv'
test_path = os.getcwd() + mainPath + 'test.csv'

# # read train csv mapping
train = pd.read_csv(train_path_short)
test = pd.read_csv(test_path)

# # split data
validate = train.sample(validationSample)
train = train.drop(validate.index)

# # refactor index on data-frames
train = train.reset_index(drop=True)
validate = validate.reset_index(drop=True)

###
# # read wav's
# validate_wavs = ut.load_by_ids(validate['ID'].values, wavPath)

# # test plot of converted wav's
# ut.plot_waves(validate['Class'].values, validate_wavs)
# ut.plot_spectrogram(validate['Class'].values, validate_wavs)
# ut.plot_log_power_spectrogram(validate['Class'].values, validate_wavs)
###

# # all featurs in a list(1) of list(2)
# # list(2) contains all feature values from the transformed signal in an np.hstack
# # one list(2) has currently a size of 193
# # after normalize np.ndarray((x,193))

validate_waves = ut.extract_by_ids(validate['ID'].values, wavPath)
validate_waves = ut.feature_normalize(validate_waves)
validate_labels = ut.one_hot_encode_list(validate['Class'].values)
# # TODO Keras one hot encoding system not working with strings
# validate_labels_2 = keras.utils.to_categorical(validate['Class'].values, len(ut.urban_class.values()))

train_waves = ut.extract_by_ids(train['ID'].values, wavPath)
train_waves = ut.feature_normalize(train_waves)
train_labels = ut.one_hot_encode_list(train['Class'].values)
# # TODO Keras one hot encoding system not working with strings
# train_labels_2 = keras.utils.to_categorical(train['Class'].values, len(ut.urban_class.values()))

test_waves = ut.extract_by_ids(test['ID'].values, wavPath)
test_waves = ut.feature_normalize(test_waves)
test_labels = ut.one_hot_encode_list(test['Class'].values)
# # TODO Keras one hot encoding system not working with strings
# train_labels_2 = keras.utils.to_categorical(train['Class'].values, len(ut.urban_class.values()))


batch_size = 30
num_classes = 10
epochs = 50

# Define MLP
model = Sequential()
model.add(Dense(512, activation='sigmoid', input_shape=(193,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# View model
model.summary()

#compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#train model
history = model.fit(train_waves, train_labels,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(validate_waves, validate_labels))

#test model
score = model.evaluate(test_waves, test_labels, verbose=0)
#print result of test
print('Test loss:', score[0])
print('Test accuracy:', score[1])