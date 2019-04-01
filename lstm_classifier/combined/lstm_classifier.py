import torch
import sys
import pickle
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils import load_data, evaluate, plot_confusion_matrix

from config import model_config as config


class LSTMClassifier(nn.Module):
    """docstring for LSTMClassifier"""
    def __init__(self, config):
        super(LSTMClassifier, self).__init__()
        self.n_layers = config['n_layers']
        self.dropout = config['dropout'] if self.n_layers > 1 else 0
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.bidirectional = config['bidirectional']

        self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, bias=True,
                           num_layers=self.n_layers, dropout=self.dropout,
                           bidirectional=self.bidirectional)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)
        self.softmax = F.softmax

    def forward(self, input_seq):
        # input_seq =. [1, batch_size, input_size]
        rnn_output, (hidden, _) = self.rnn(input_seq)
        if self.bidirectional:  # sum outputs from the two directions
            rnn_output = rnn_output[:, :, :self.hidden_dim] +\
                        rnn_output[:, :, self.hidden_dim:]
        class_scores = F.softmax(self.out(rnn_output[0]), dim=1)
        return class_scores


if __name__ == '__main__':
    emotion_dict = {'ang': 0, 'hap': 1, 'sad': 2, 'fea': 3, 'sur': 4, 'neu': 5}

    device = 'cuda:{}'.format(config['gpu']) if \
             torch.cuda.is_available() else 'cpu'

    model = LSTMClassifier(config)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    train_batches = load_data()
    test_pairs = load_data(test=True)

    best_acc = 0
    for epoch in range(config['n_epochs']):
        losses = []
        for batch in train_batches:
            inputs = batch[0].unsqueeze(0)  # frame in format as expected by model
            targets = batch[1]
            inputs = inputs.to(device)
            targets = targets.to(device)

            model.zero_grad()
            optimizer.zero_grad()

            predictions = model(inputs)
            predictions = predictions.to(device)

            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # evaluate
        with torch.no_grad():
            inputs = test_pairs[0].unsqueeze(0)
            targets = test_pairs[1]

            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = torch.argmax(model(inputs), dim=1)  # take argmax to get class id
            predictions = predictions.to(device)

            # evaluate on cpu
            targets = np.array(targets.cpu())
            predictions = np.array(predictions.cpu())

            # Get results
            # plot_confusion_matrix(targets, predictions,
            #                       classes=emotion_dict.keys())
            performance = evaluate(targets, predictions)
            if performance['acc'] > best_acc:
                print(performance)
                best_acc = performance['acc']
                # save model and results
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, 'runs/{}-best_model.pth'.format(config['model_code']))

                with open('results/{}-best_performance.pkl'.format(config['model_code']), 'wb') as f:
                    pickle.dump(performance, f)
