import pickle
import numpy as np
import pandas as pd


class Vocabulary(object):
    def __init__(self):
        self.word2index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.word2count = {}
        self.index2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.size = 4  # Count PAD, SOS, EOS and UNK

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.size
            self.word2count[word] = 1
            self.index2word[self.size] = word
            self.size += 1
        else:
            self.word2count[word] += 1


def create_vocab(file_dir='../../data/t2e/'):
    print('Loading corpus...')
    texts = []
    for mode in ['train', 'test']:
        texts += list(pd.read_csv('{}text_{}.csv'.format(file_dir, mode))['transcription'])

    print("Building vocab...")
    vocab = Vocabulary()

    for text in texts:
        vocab.add_sentence(text)

    print("Total words in vocab:  {}".format(vocab.size))
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    print('Generating word embeddings')
    return vocab


if __name__ == '__main__':
    create_vocab()
