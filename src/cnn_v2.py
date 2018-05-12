import pre
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling2D

# # global variables
validationSample = 240
wavSource = False
data_type = 'cnn'

train_waves, train_labels, validate_waves, validate_labels, test_waves, test_labels = \
    pre.load(wavSource, validationSample, data_type)

batch_size = 30
num_classes = 10
epochs = 50

print(train_waves.shape)
print(train_waves.min())
print(train_waves.max())

input_shape = (60, 41, 2)

train_waves = train_waves.astype('float32')
validate_waves = validate_waves.astype('float32')
test_waves = test_waves.astype('float32')

#CNN to train mnist dataset
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

model.fit(train_waves, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(validate_waves, validate_labels))

score = model.evaluate(test_waves, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
