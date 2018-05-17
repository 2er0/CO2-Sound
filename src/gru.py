import os

import after
import pre
from keras.models import Sequential
from keras.layers import Dense, GRU

# # global variables
validationSample = 240
wavSource = False
data_type = 'lstm'
input_shape = (41, 20)

# validate_waves, validate_labels,
train_waves, train_labels, test_waves, test_labels = \
    pre.load(input_shape, data_type)

batch_size = 30
num_classes = 10
epochs = 50

print(train_waves[0].shape)
print(train_waves.shape, train_labels.shape)

# Define LSTM
model = Sequential()
model.add(GRU(100, input_shape=input_shape))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# compile model
model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy
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

after.saveModel(model, os.path.basename(__file__))
