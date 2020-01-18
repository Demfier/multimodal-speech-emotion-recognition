"""
This script should be run AFTER extract_emotion_labels.py. It basically uses
the csv file created in the previous step to split the original wav files into
multiple smaller frames, each piece containing an emotion.

Run this script from root as python src/build_audio_vectors.py
"""

import os
import math
import pickle
import librosa
import pandas as pd
from tqdm import tqdm


def process_session(iemocap_dir, labels_df, sr, sess):
    """
    saves audio_vectors dict in a pickle file which contains vectors
    for audio files in session `sess`

    process_session: Str pd.DataFrame Nat Int -> None
    """
    audio_vectors = {}
    wav_file_path = '{}Session{}/dialog/wav/'.format(iemocap_dir, sess)
    orig_wav_files = os.listdir(wav_file_path)
    for orig_wav_file in tqdm(orig_wav_files):
        try:
            orig_wav_vector, _sr = librosa.load(wav_file_path + orig_wav_file,
                                                sr=sr)
            orig_wav_file, file_format = orig_wav_file.split('.')
            for index, row in labels_df[labels_df['wav_file'].str.contains(
                    orig_wav_file)].iterrows():
                start_time, end_time, truncated_wav_file_name, = \
                    row['start_time'], row['end_time'], row['wav_file']
                start_frame = math.floor(start_time * sr)
                end_frame = math.floor(end_time * sr)
                truncated_wav_vector = orig_wav_vector[start_frame:end_frame+1]
                audio_vectors[truncated_wav_file_name] = truncated_wav_vector
        except Exception as e:
            print('An exception occured for {}'.format(orig_wav_file))
    with open('data/pre-processed/audio_vectors_{}.pkl'.format(sess), 'wb') as f:
        pickle.dump(audio_vectors, f)


def main():
    sampling_rate = 44100
    iemocap_dir = 'data/IEMOCAP_full_release/'
    labels_df = pd.read_csv('data/pre-processed/df_iemocap.csv')
    for sess in range(1, 6):
        # Note that compiling this way will take too much time So you might
        # consider parallelizing this process
        process_session(iemocap_dir, labels_df, sampling_rate, sess)


if __name__ == '__main__':
    main()
