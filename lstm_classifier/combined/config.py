model_config = {
    'gpu': 1,
    'n_layers': 2,
    'dropout': 0.2,
    'output_dim': 6,  # number of classes
    'hidden_dim': 256,
    'input_dim': 2472,
    'batch_size': 200,  # carefully chosen
    'n_epochs': 55000,
    'learning_rate': 0.001,
    'bidirectional': True,
    'model_code': 'bi_lstm'
}
