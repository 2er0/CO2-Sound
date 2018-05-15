import matplotlib.pyplot as plt

# history,
def plotAll(history, score, epochs: int, name: str):
    name = name.split('.')[0]

    # summarize history for accuracy
    accLast = history.history['acc'][-1]
    valAccLast = history.history['val_acc'][-1]

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])

    plt.axhline(y=accLast, color='grey')
    plt.annotate(accLast, xy=(0, accLast), bbox=dict(boxstyle="round4", fc="w"))
    plt.axhline(y=valAccLast, color='grey')
    plt.annotate(valAccLast, xy=(0, valAccLast), bbox=dict(boxstyle="round4", fc="w"))

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

    plt.axhline(y=lossLast, color='grey')
    plt.annotate(lossLast, xy=(0, lossLast), bbox=dict(boxstyle="round4", fc="w"))
    plt.axhline(y=valLossLast, color='grey')
    plt.annotate(valLossLast, xy=(0, valLossLast), bbox=dict(boxstyle="round4", fc="w"))

    plt.title('{} model loss - testLoss {}'.format(name, score[0]))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='center right')
    plt.draw()
    plt.savefig('../data/results/{}_loss.png'.format(name))