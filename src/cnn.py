import os

import after
import pre
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# global variables
validationSample = 240
wavSource = False
data_type = 'cnn'

input_shape = (60, 41, 2)

# validate_waves, validate_labels,
train_waves, train_labels, test_waves, test_labels = \
    pre.load(input_shape, data_type)

batch_size = 250
num_classes = 10
epochs = 25

print(train_waves[0].shape)
print(train_waves.shape, train_labels.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_waves, train_labels,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.15,
                    shuffle=True)

score = model.evaluate(test_waves, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# generate plots und store them
after.plotAll(history, score, epochs, os.path.basename(__file__))
