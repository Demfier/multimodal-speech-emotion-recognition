import os
import re
import math
import random
import pickle
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf


# Part 1: Extract Audio Labels
def extract_audio_labels():
    info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)

    start_times, end_times, wav_file_names, emotions, vals, acts, doms = \
        [], [], [], [], [], [], []

    for sess in range(1, 6):
        emo_evaluation_dir = \
            'data/IEMOCAP_full_release/Session{}/dialog/EmoEvaluation/'.format(sess)
        evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]
        for file in evaluation_files:
            with open(emo_evaluation_dir + file) as f:
                content = f.read()
            info_lines = re.findall(info_line, content)
            for line in info_lines[1:]:  # the first line is a header
                start_end_time, wav_file_name, emotion, val_act_dom = \
                    line.strip().split('\t')
                start_time, end_time = start_end_time[1:-1].split('-')
                val, act, dom = val_act_dom[1:-1].split(',')
                val, act, dom = float(val), float(act), float(dom)
                start_time, end_time = float(start_time), float(end_time)
                start_times.append(start_time)
                end_times.append(end_time)
                wav_file_names.append(wav_file_name)
                emotions.append(emotion)
                vals.append(val)
                acts.append(act)
                doms.append(dom)

    df_iemocap = pd.DataFrame(columns=['start_time', 'end_time', 'wav_file',
                                       'emotion', 'val', 'act', 'dom'])

    df_iemocap['start_time'] = start_times
    df_iemocap['end_time'] = end_times
    df_iemocap['wav_file'] = wav_file_names
    df_iemocap['emotion'] = emotions
    df_iemocap['val'] = vals
    df_iemocap['act'] = acts
    df_iemocap['dom'] = doms

    df_iemocap.to_csv('data/pre-processed/df_iemocap.csv', index=False)


# Part 2: Build Audio Vectors
def build_audio_vectors():
    labels_df = pd.read_csv('data/pre-processed/df_iemocap.csv')
    iemocap_dir = 'data/IEMOCAP_full_release/'

    sr = 44100
    audio_vectors = {}
    for sess in range(1, 6):  # using one session due to memory constraint, can replace [5] with range(1, 6)
        wav_file_path = '{}Session{}/dialog/wav/'.format(iemocap_dir, sess)
        orig_wav_files = os.listdir(wav_file_path)
        for orig_wav_file in tqdm(orig_wav_files):
            try:
                orig_wav_vector, _sr = librosa.load(
                        wav_file_path + orig_wav_file, sr=sr)
                orig_wav_file, file_format = orig_wav_file.split('.')
                for index, row in labels_df[labels_df['wav_file'].str.contains(
                        orig_wav_file)].iterrows():
                    start_time, end_time, truncated_wav_file_name, emotion,\
                        val, act, dom = row['start_time'], row['end_time'],\
                        row['wav_file'], row['emotion'], row['val'],\
                        row['act'], row['dom']
                    start_frame = math.floor(start_time * sr)
                    end_frame = math.floor(end_time * sr)
                    truncated_wav_vector = orig_wav_vector[start_frame:end_frame + 1]
                    audio_vectors[truncated_wav_file_name] = truncated_wav_vector
            except:
                print('An exception occured for {}'.format(orig_wav_file))
        with open('data/pre-processed/audio_vectors_{}.pkl'.format(sess), 'wb') as f:
            pickle.dump(audio_vectors, f)


# Part 3: Extract Audio Features
def extract_audio_features():
    data_dir = 'data/pre-processed/'
    labels_df_path = '{}df_iemocap.csv'.format(data_dir)
    audio_vectors_path = '{}audio_vectors_1.pkl'.format(data_dir)
    labels_df = pd.read_csv(labels_df_path)
    audio_vectors = pickle.load(open(audio_vectors_path, 'rb'))

    columns = ['wav_file', 'label', 'sig_mean', 'sig_std', 'rmse_mean',
               'rmse_std', 'silence', 'harmonic', 'auto_corr_max', 'auto_corr_std']
    df_features = pd.DataFrame(columns=columns)

    emotion_dict = {'ang': 0,
                    'hap': 1,
                    'exc': 2,
                    'sad': 3,
                    'fru': 4,
                    'fea': 5,
                    'sur': 6,
                    'neu': 7,
                    'xxx': 8,
                    'oth': 8}

    data_dir = 'data/pre-processed/'
    labels_path = '{}df_iemocap.csv'.format(data_dir)
    audio_vectors_path = '{}audio_vectors_'.format(data_dir)
    labels_df = pd.read_csv(labels_path)

    for sess in (range(1, 6)):
        audio_vectors = pickle.load(open('{}{}.pkl'.format(audio_vectors_path, sess), 'rb'))
        for index, row in tqdm(labels_df[labels_df['wav_file'].str.contains('Ses0{}'.format(sess))].iterrows()):
            try:
                wav_file_name = row['wav_file']
                label = emotion_dict[row['emotion']]
                y = audio_vectors[wav_file_name]

                feature_list = [wav_file_name, label]  # wav_file, label
                sig_mean = np.mean(abs(y))
                feature_list.append(sig_mean)  # sig_mean
                feature_list.append(np.std(y))  # sig_std

                rmse = librosa.feature.rmse(y + 0.0001)[0]
                feature_list.append(np.mean(rmse))  # rmse_mean
                feature_list.append(np.std(rmse))  # rmse_std

                silence = 0
                for e in rmse:
                    if e <= 0.4 * np.mean(rmse):
                        silence += 1
                silence /= float(len(rmse))
                feature_list.append(silence)  # silence

                y_harmonic = librosa.effects.hpss(y)[0]
                feature_list.append(np.mean(y_harmonic) * 1000)  # harmonic (scaled by 1000)

                # based on the pitch detection algorithm mentioned here:
                # http://access.feld.cvut.cz/view.php?cisloclanku=2009060001
                cl = 0.45 * sig_mean
                center_clipped = []
                for s in y:
                    if s >= cl:
                        center_clipped.append(s - cl)
                    elif s <= -cl:
                        center_clipped.append(s + cl)
                    elif np.abs(s) < cl:
                        center_clipped.append(0)
                auto_corrs = librosa.core.autocorrelate(np.array(center_clipped))
                feature_list.append(1000 * np.max(auto_corrs)/len(auto_corrs))  # auto_corr_max (scaled by 1000)
                feature_list.append(np.std(auto_corrs))  # auto_corr_std

                df_features = df_features.append(pd.DataFrame(
                    feature_list, index=columns).transpose(),
                    ignore_index=True)
            except:
                print('Some exception occured')

    df_features.to_csv('data/pre-processed/audio_features.csv', index=False)


def main():
    print('Part 1: Extract Audio Labels')
    extract_audio_labels()
    print('Part 2: Build Audio Vectors')
    build_audio_vectors()
    print('Part 3: Extract Audio Features')
    extract_audio_features()


if __name__ == '__main__':
    main()
