import pre
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D

# # global variables
validationSample = 340
wavSource = False
data_type = 'nn'
load_shape = (193, )

# validate_waves, validate_labels,
train_waves, train_labels, test_waves, test_labels = \
    pre.load(load_shape, data_type)

batch_size = 30
num_classes = 10
epochs = 50

train_waves = train_waves[:, :, np.newaxis]
test_waves = test_waves[:, :, np.newaxis]

print(train_waves[0].shape)
print(train_waves.shape, train_labels.shape)
print(test_waves.shape)

input_shape = (193, 1)

#CNN with conv1D dataset
model = Sequential()
model.add(Conv1D(32, kernel_size=3,
                 activation='relu',
                 input_shape=input_shape))

model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_waves, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.15)
          #validation_data=(validate_waves, validate_labels))

score = model.evaluate(test_waves, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
