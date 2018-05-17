import os

import pre
import after
from keras.models import Sequential
from keras.layers import Dense, Dropout

# # global variables
validationSample = 340
wavSource = False
data_type = 'nn8k'
input_shape = (193,)

# validate_waves, validate_labels,
train_waves, train_labels, test_waves, test_labels = \
    pre.load8k(input_shape, data_type)

batch_size = 30
num_classes = 10
epochs = 50

print(train_waves.shape)
print(test_waves.shape)

# Define MLP
model = Sequential()
model.add(Dense(512, activation='sigmoid', input_shape=input_shape))
model.add(Dropout(0.2))
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# View model
model.summary()

# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# train model
history = model.fit(train_waves, train_labels,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.15,
                    shuffle=True)

# test model
score = model.evaluate(test_waves, test_labels, verbose=0)
# print result of test
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# generate plots und store them
after.plotAll(history, score, epochs, os.path.basename(__file__))
