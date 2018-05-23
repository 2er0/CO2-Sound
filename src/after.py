import itertools
import numpy as np
import matplotlib.pyplot as plt

import utils as ut

# history,
def plotAll(history, score, epochs: int, name: str):
    name = name.split('.')[0]

    # summarize history for accuracy
    accLast = history.history['acc'][-1]
    valAccLast = history.history['val_acc'][-1]

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])

    plt.axhline(y=accLast, color='grey', alpha=0.5)
    plt.annotate("{0:.4f}".format(accLast), xy=(0, accLast), bbox=dict(boxstyle="round4", fc="w", alpha=0.5))
    plt.axhline(y=valAccLast, color='grey', alpha=0.5)
    plt.annotate("{0:.4f}".format(valAccLast), xy=(int(epochs/5), valAccLast), bbox=dict(boxstyle="round4", fc="w", alpha=0.5))

    plt.title('{} model accuracy - testAcc {}'.format(name, score[1]))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='center right')
    plt.draw()
    plt.savefig('../data/results/{}_acc.png'.format(name))

    # summarize history for loss
    lossLast = history.history['loss'][-1]
    valLossLast = history.history['val_loss'][-1]

    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.axhline(y=lossLast, color='grey', alpha=0.5)
    plt.annotate("{0:.4f}".format(lossLast), xy=(0, lossLast), bbox=dict(boxstyle="round4", fc="w", alpha=0.5))
    plt.axhline(y=valLossLast, color='grey', alpha=0.5)
    plt.annotate("{0:.4f}".format(valLossLast), xy=(int(epochs/5), valLossLast), bbox=dict(boxstyle="round4", fc="w", alpha=0.5))

    plt.title('{} model loss - testLoss {}'.format(name, score[0]))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='center right')
    plt.draw()
    plt.savefig('../data/results/{}_loss.png'.format(name))


# save model to file
def saveModel(model, name: str):
    name = name.split('.')[0]
    model.save('../data/models/{}.h5'.format(name))


# generate confMatrix
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, count, title,
                          normalize=True,
                          cmap=plt.cm.Blues):

    classes = ut.urban_class.keys()
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.clf()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion matrix | " + str(count) + " | " + title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.draw()
    plt.savefig('../data/results/{}_matrix.png'.format(title))
