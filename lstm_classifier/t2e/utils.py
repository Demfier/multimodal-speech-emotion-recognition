import os
import torch
import pickle
import gensim
import numpy as np
import pandas as pd
from config import model_config as config
from gensim.scripts.glove2word2vec import glove2word2vec

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

import itertools
import matplotlib.pyplot as plt


def generate_word_embeddings(vocab):
    if not os.path.exists('{}gensim.glove.6B.{}d.txt'.format(
            config['embeddings_dir'], config['embedding_dim'])):
        glove2word2vec(glove_input_file='{}glove.6B.{}d.txt'.format(
            config['embeddings_dir'], config['embedding_dim']),
            word2vec_output_file='{}gensim.glove.6B.{}d.txt'.format(
            config['embeddings_dir'], config['embedding_dim']))

    embeddings_all = gensim.models.KeyedVectors.load_word2vec_format(
        '{}gensim.glove.6B.{}d.txt'.format(config['embeddings_dir'],
                                           config['embedding_dim']))
    print('Loaded original embeddings')

    # initialize word embeddings matrix
    combined_word_embeddings = np.zeros((vocab.size,
                                         config['embedding_dim']))
    for index, word in vocab.index2word.items():
        try:
            if index < 4:  # deal with special tokens
                combined_word_embeddings[index] = np.random.normal(
                    size=(config['embedding_dim'], ))
                continue
            combined_word_embeddings[index] = embeddings_all[word]
        except KeyError as e:
            print('KeyError triggered for {}'.format(word))
            combined_word_embeddings[index] = np.random.normal(
                size=(config['embedding_dim'], ))
    print('Created combined + filtered embeddings.')
    with open('{}saved_{}d_word_embeddings.pkl'.format(
            config['embeddings_dir'], config['embedding_dim']), 'wb') as f:
        pickle.dump(combined_word_embeddings, f)
    combined_word_embeddings = torch.from_numpy(combined_word_embeddings).float()
    return combined_word_embeddings


def load_word_embeddings():
    with open('{}saved_{}d_word_embeddings.pkl'.format(
            config['embeddings_dir'], config['embedding_dim']), 'rb') as f:
        combined_word_embeddings = pickle.load(f)
        return torch.from_numpy(combined_word_embeddings).float()


def zero_padding(l, fillvalue=config['<PAD>']):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binary_matrix(l, value=config['<PAD>']):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == 0:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Returns padded input sequence tensor and lengths
def input_var(l, vocab):
    indexes_batch = [indexes_from_sentence(vocab, sentence) for sentence in l]
    for idx, indexes in enumerate(indexes_batch):
        indexes_batch[idx] = indexes_batch[idx] + [config['<EOS>']]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    pad_list = zero_padding(indexes_batch)
    pad_var = torch.LongTensor(pad_list)
    return pad_var, lengths


def indexes_from_sentence(vocab, sentence):
    indexes = []
    for word in sentence.strip().split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError as e:
            indexes.append(config['<UNK>'])
    return indexes[:config['max_sequence_length']]


def load_data(batched=True, test=False, file_dir='../../data/t2e/'):
    # Load vocab
    with open(config['vocab_path'], 'rb') as f:
        vocab = pickle.load(f)

    bs = config['batch_size']
    ftype = 'test' if test else 'train'

    df = pd.read_csv('{}text_{}.csv'.format(file_dir, ftype))
    data = (np.array(list(df['transcription'])), np.array(df['label']))

    data = list(zip(data[0], data[1]))
    data.sort(key=lambda x: len(x[0].split()), reverse=True)

    n_iters = len(data) // bs

    if test:
        input_batch = []
        output_batch = []
        for e in data:
            input_batch.append(e[0])
            output_batch.append(e[1])
        inp, lengths = input_var(input_batch, vocab)
        return [inp, lengths, torch.LongTensor(output_batch)]

    batches = []
    for i in range(1, n_iters + 1):
        input_batch = []
        output_batch = []
        for e in data[bs * (i-1):bs * i]:
            input_batch.append(e[0])
            output_batch.append(e[1])
        inp, lengths = input_var(input_batch, vocab)
        batches.append([inp, lengths,
                        torch.LongTensor(output_batch)])
    return batches


def evaluate(targets, predictions):
    performance = {
        'acc': accuracy_score(targets, predictions),
        'f1': f1_score(targets, predictions, average='macro'),
        'precision': precision_score(targets, predictions, average='macro'),
        'recall': recall_score(targets, predictions, average='macro')}
    return performance


def plot_confusion_matrix(targets, predictions, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # plt.figure(figsize=(8,8))
    cm = confusion_matrix(targets, predictions)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
