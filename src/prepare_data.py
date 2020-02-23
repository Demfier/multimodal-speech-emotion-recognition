"""
This script preprocesses data and prepares data to be actually used in training
"""
import re
import os
import pickle
import unicodedata
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    """
    Lowercase, trim, and remove non-letter characters
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def transcribe_sessions():
    file2transcriptions = {}
    useful_regex = re.compile(r'^(\w+)', re.IGNORECASE)
    transcript_path = 'data/IEMOCAP_full_release/Session{}/dialog/transcriptions/'
    for sess in range(1, 6):
        transcript_path = transcript_path.format(sess)
        for f in os.listdir(transcript_path):
            with open('{}{}'.format(transcript_path, f), 'r') as f:
                all_lines = f.readlines()

            for l in all_lines:
                audio_code = useful_regex.match(l).group()
                transcription = l.split(':')[-1].strip()
                # assuming that all the keys would be unique and hence no `try`
                file2transcriptions[audio_code] = transcription
    with open('data/t2e/audiocode2text.pkl', 'wb') as file:
        pickle.dump(file2transcriptions, file)
    return file2transcriptions


def prepare_text_data(audiocode2text):
    # Prepare text data
    text_train = pd.DataFrame()
    text_train['wav_file'] = x_train['wav_file']
    text_train['label'] = x_train['label']
    text_train['transcription'] = [normalizeString(audiocode2text[code])
                                   for code in x_train['wav_file']]

    text_test = pd.DataFrame()
    text_test['wav_file'] = x_test['wav_file']
    text_test['label'] = x_test['label']
    text_test['transcription'] = [normalizeString(audiocode2text[code])
                                  for code in x_test['wav_file']]

    text_train.to_csv('data/t2e/text_train.csv', index=False)
    text_test.to_csv('data/t2e/text_test.csv', index=False)

    print(text_train.shape, text_test.shape)


def main():
    prepare_text_data(transcribe_sessions())


if __name__ == '__main__':
    main()
