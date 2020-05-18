import torch
import pickle
import gensim
from create_vocab import Vocabulary, create_vocab

model_config = {
    'gpu': 1,
    '<PAD>': 0,
    '<SOS>': 1,
    '<EOS>': 2,
    '<UNK>': 3,
    'n_layers': 2,
    'dropout': 0.2,
    'output_dim': 6,  # number of classes
    'hidden_dim': 500,
    'n_epochs': 45000,
    'batch_size': 128,  # carefully chosen
    'embedding_dim': 200,  # 50/100/200/300
    'bidirectional': True,
    'learning_rate': 0.0001,
    'model_code': 'bi_lstm_2_layer',
    'max_sequence_length': 20,
    'embeddings_dir': 'embeddings/'
}


from utils import generate_word_embeddings


def set_dynamic_hparams():
    try:
        with open('vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
    except FileNotFoundError as e:
        vocab = create_vocab()
        generate_word_embeddings(vocab)

    model_config['vocab_size'] = vocab.size
    model_config['vocab_path'] = 'vocab.pkl'
    return model_config


model_config = set_dynamic_hparams()
