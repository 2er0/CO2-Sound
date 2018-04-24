import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram

plt.style.use('ggplot')

plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 13


def load_sounds(paths):
    raw_sounds = []
    for p in paths:
        x, _ = librosa.load(p)
        raw_sounds.append(x)
    return raw_sounds


def load_by_id(ids: list, path: str):
    wav_files = list(map(lambda x: path + str(x) + '.wav', ids))
    return load_sounds(wav_files)


def plot_waves(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25, 60), dpi=300)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(5, 1, i)
        librosa.display.waveplot(np.array(f), sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 1: Waveplot', x=0.5, y=0.915, fontsize=18)
    plt.savefig('../data/waves.png')


def plot_spectrogram(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25, 60), dpi=300)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(5, 1, i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 2: Spectrogram', x=0.5, y=0.915, fontsize=18)
    plt.savefig('../data/specgram.png')


def plot_log_power_spectrogram(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25, 60), dpi=300)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(5, 1, i)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(f)) ** 2, ref=np.max)
        librosa.display.specshow(D, x_axis='time', y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 3: Log power spectrogram', x=0.5, y=0.915, fontsize=18)
    plt.savefig('../data/log_power_specgram.png')
