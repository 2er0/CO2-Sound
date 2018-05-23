# Import the modules from `sklearn.metrics`
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from keras.models import Sequential
from keras.layers import Dense, Dropout

import pre
import after
import utils as ut

# # global variables
data_type = 'nn'
input_shape = (193,)
num_classes = 10

# validate_waves, validate_labels,
train_waves, train_labels, test_waves, test_labels = \
    pre.load(input_shape, data_type)

model = Sequential()
model.add(Dense(512, activation='sigmoid', input_shape=input_shape))
model.add(Dropout(0.2))
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.load_weights('../data/models/nn.h5')

test_labels_pred = model.predict(test_waves)
test_labels_pred = test_labels_pred.argmax(axis=-1)
test_labels = test_labels.argmax(axis=-1)

# Confusion matrix
confmatrix = confusion_matrix(test_labels, test_labels_pred)

after.plot_confusion_matrix(confmatrix, "tmp")
