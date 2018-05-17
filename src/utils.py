import librosa
import librosa.display
import numpy as np
import pandas as pd
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


# create file path to wav
def build_file_path(i: int, p: str):
    return p + str(i) + '.wav'


# create file path to 8k wav
def build_file_path_8k(i: str, f: str, p:str):
    return p + 'fold' + str(f) + '/' + i


# load sounds from paths to ndarray
def load_sounds(paths) -> np.ndarray:
    raw_sounds = np.empty((0,))
    for p in paths:
        x, _ = librosa.load(p)
        raw_sounds = np.vstack([raw_sounds, x])
    return raw_sounds


# load sound by id and main path
def load_by_id(i: int, p: str) -> np.ndarray:
    return load_sounds([build_file_path(i, p)])


# load sounds by ids and main path
def load_by_ids(ids: list, p: str) -> list:
    wav_files = list(map(lambda i: load_by_id(i, p), ids))
    return wav_files


# extract most possible data from sound by id and path
# NN/MLP
def extract_by_id_full(i: int, p: str) -> np.ndarray:
    return extract_feature_full(build_file_path(i, p))


# extract most possible data from sounds by ids and path
# NN/MLP
def extract_by_ids_full(ids: list, p: str) -> list:
    wav_files = list(map(lambda i: extract_by_id_full(i, p), ids))
    return wav_files


# extract from 8k for nn
def extract_by_ids_full_8k(data: pd.DataFrame, p: str) -> list:
    res = list()

    for row in data.iterrows():
        sound = extract_feature_full(build_file_path_8k(row[1]['slice_file_name'], row[1]['fold'], p))
        if len(sound) < 1:
            continue

        label = one_hot_encode(row[1]['class'])

        res.append((sound, label))

    return res


# extract most possible data from sound by id and path
# CNN
def extract_by_id_cnn(i: int, p: str) -> np.ndarray:
    return extract_feature_cnn(build_file_path(i, p))


# extract most possible data from sounds by ids and path
# CNN
def extract_by_ids_cnn(ids: list, p: str) -> list:
    return list(map(lambda i: extract_by_id_cnn(i, p), ids))


# extract from 8k for cnn
def extract_by_ids_cnn_8k(data: pd.DataFrame, p: str) -> list:
    res = list()

    for row in data.iterrows():
        sound = extract_feature_cnn(build_file_path_8k(row[1]['slice_file_name'], row[1]['fold'], p))
        if len(sound) < 1:
            continue

        label = one_hot_encode(row[1]['class'])

        res.append((sound, label))

    return res


# extract most possible data from sounds by id and path
# LSTM
def extract_by_id_lstm(i: int, p: str) -> np.ndarray:
    return extract_feature_lstm(build_file_path(i, p))


# extract most possible data from sounds by ids and path
# LSTM
def extract_by_ids_lstm(ids: list, p: str) -> list:
    return list(map(lambda i: extract_by_id_lstm(i, p), ids))


# extract from 8k for lstm
def extract_by_ids_lstm_8k(data: pd.DataFrame, p: str) -> list:
    res = list()

    for row in data.iterrows():
        sound = extract_feature_lstm(build_file_path_8k(row[1]['slice_file_name'], row[1]['fold'], p))
        if len(sound) < 1:
            continue

        label = one_hot_encode(row[1]['class'])

        res.append((sound, label))

    return res


# one hot encode by given label
def one_hot_encode(label: str) -> np.ndarray:
    vec = np.zeros(10)
    vec[urban_class[label]] = 1
    return vec


# one hot encode by given labels
def one_hot_encode_list(ls: list) -> np.ndarray:
    encoded = np.empty((0, len(urban_class.values())))
    for l in ls:
        encoded = np.vstack([encoded, one_hot_encode(l)])
    return np.array(encoded)


# normalize features to float32
def feature_normalize(ls: list) -> np.ndarray:
    ls = np.array(ls)
    ls = ls.astype('float32')
    return ls


# plot stop criteria
stop = 5


# some plot tool
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


# extract most possible data from filename name
def extract_feature_full(file_name):
    try:
        X, sample_rate = librosa.load(file_name)
        stft = np.abs(librosa.stft(X))
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
        return np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    except librosa.ParameterError:
        print(file_name)


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2)


# extract most possible data from filename for cnn
def extract_feature_cnn(file_name, bands=60, frames=41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    sound_clip, sample_rate = librosa.load(file_name)
    for (start, end) in windows(sound_clip, window_size):
        if (len(sound_clip[start:end]) == window_size):
            signal = sound_clip[start:end]
            melspec = librosa.feature.melspectrogram(signal, sr=sample_rate, n_mels=bands)
            logspec = librosa.amplitude_to_db(melspec)
            logspec = logspec.T.flatten()
            log_specgrams.append(logspec)

    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

    return np.array(features)


# extract most possible data from filename for lstm
def extract_feature_lstm(file_name, bands=20, frames=41):
    window_size = 512 * (frames - 1)
    mfccs = []
    sound_clip, sample_rate = librosa.load(file_name)
    for (start, end) in windows(sound_clip, window_size):
        if (len(sound_clip[start:end]) == window_size):
            signal = sound_clip[start:end]
            mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=bands).T.flatten()
            mfccs.append(mfcc)

    features = np.asarray(mfccs).reshape(len(mfccs), frames, bands)
    return np.array(features)
