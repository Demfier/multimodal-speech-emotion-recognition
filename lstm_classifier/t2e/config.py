import pickle
from create_vocab import create_vocab

model_config = {
    'gpu': 1,
    '<PAD>': 0,
    '<SOS>': 1,
    '<EOS>': 2,
    '<UNK>': 3,
    'n_layers': 3,
    'dropout': 0.2,
    'output_dim': 6,  # number of classes
    'hidden_dim': 50,
    'n_epochs': 45000,
    'batch_size': 200,  # carefully chosen
    'embedding_dim': 50,
    'learning_rate': 0.001,
    'bidirectional': True,
    'max_sequence_length': 20,
    'model_code': 'bi_lstm'
}


def set_dynamic_hparams():
    try:
        with open('vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
    except FileNotFoundError as e:
        vocab = create_vocab()

    model_config['vocab_size'] = vocab.size
    model_config['vocab_path'] = 'vocab.pkl'
    return model_config


model_config = set_dynamic_hparams()
