import pre
from keras.models import Sequential
from keras.layers import Dense, LSTM

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
epochs = 5

print(train_waves[0].shape)
print(train_waves.shape, train_labels.shape)

# Define LSTM
model = Sequential()
model.add(LSTM(100, input_shape=input_shape))
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
                    validation_split=0.15)
                    #validation_data=(validate_waves, validate_labels))

# test model
score = model.evaluate(test_waves, test_labels, verbose=0)
# print result of test
print('Test loss:', score[0])
print('Test accuracy:', score[1])
