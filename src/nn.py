import pre
from keras.models import Sequential
from keras.layers import Dense, Dropout

# # global variables
validationSample = 30
wavSource = False

train_waves, train_labels, validate_waves, validate_labels, test_waves, test_labels = \
    pre.load(wavSource, validationSample)

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

# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# train model
history = model.fit(train_waves, train_labels,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(validate_waves, validate_labels))

# test model
score = model.evaluate(test_waves, test_labels, verbose=0)
# print result of test
print('Test loss:', score[0])
print('Test accuracy:', score[1])
