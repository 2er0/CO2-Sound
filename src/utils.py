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

build_file_path = lambda i, p: p + str(i) + '.wav'


urban_class = {
    'air_conditioner': 0,
    'car_horn': 1,
    'children_playing': 2,
    'dog_bark': 3,
    'drilling': 4,
    'engine_idling': 5,
    'gun_shot': 6,
    'jackhammer': 7,
    'siren': 8,
    'street_music': 9
}


def load_sounds(paths):
    raw_sounds = []
    for p in paths:
        x, _ = librosa.load(p)
        raw_sounds.append(x)
    return raw_sounds


def load_by_id(i: int, p: str):
    return load_sounds([build_file_path(i, p)])


def load_by_ids(ids: list, p: str):
    wav_files = list(map(lambda i: load_by_id(i, p), ids))
    return wav_files


def extract_by_id(i: int, p: str):
    return extract_feature(build_file_path(i, p))


def extract_by_ids(ids: list, p: str):
    wav_files = list(map(lambda i: extract_by_id(i, p), ids))
    return wav_files


def one_hot_encode(label):
    vec = np.zeros(10)
    vec[urban_class[label]] = 1
    return vec


# plot stop criteria
stop = 5


def plot_waves(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25, 60), dpi=300)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(stop, 1, i)
        librosa.display.waveplot(np.array(f), sr=22050)
        plt.title(n.title())
        i += 1
        if i >= stop:
            break
    plt.suptitle('Figure 1: Waveplot', x=0.5, y=0.915, fontsize=18)
    plt.savefig('../data/waves.png')


def plot_spectrogram(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25, 60), dpi=300)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(stop, 1, i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
        if i >= stop:
            break
    plt.suptitle('Figure 2: Spectrogram', x=0.5, y=0.915, fontsize=18)
    plt.savefig('../data/specgram.png')


def plot_log_power_spectrogram(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25, 60), dpi=300)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(stop, 1, i)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(f)) ** 2, ref=np.max)
        librosa.display.specshow(D, x_axis='time', y_axis='log')
        plt.title(n.title())
        i += 1
        if i >= stop:
            break
    plt.suptitle('Figure 3: Log power spectrogram', x=0.5, y=0.915, fontsize=18)
    plt.savefig('../data/log_power_specgram.png')


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return np.hstack([mfccs, chroma, mel, contrast, tonnetz])
