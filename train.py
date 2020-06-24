# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 17:01:50 2020

@author: 13758
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from collections import OrderedDict, defaultdict
from torch.utils.data import Dataset, DataLoader, Subset
from keras.preprocessing.sequence import pad_sequences
from data_sampler import data_infor
from pre_processing import pre_processing
from transformers import BertModel
from model_lstm_bert import bert_lstm
import argparse

def str2bool(string):
    if isinstance(string, bool):
        return string
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')
    

parser = argparse.ArgumentParser(
        description = 'Sentiment analysis training with BERT&LSTM'
        )
parser.add_argument('--freeze',
                    help = 'Freeze BERT or not',
                    type = str2bool,
                    default = True)
parser.add_argument('--nlayer',
                    help = 'The number of LSTM layers',
                    type = int,
                    default = 2)
parser.add_argument('--data',
                    help = 'The applied dataset',
                    default = 'IMDB')
parser.add_argument('--kept_prob_dropout',
                    help = 'The probability to keep params',
                    default = 1)
parser.add_argument('--learning_rate',
                    help = 'learning rate',
                    type = float,
                    default = 0.0005)
parser.add_argument('--save_path',
                    help = 'Save path',
                    default = '/content/drive/My Drive/Master_Final_Project/Genetic_attack/Code/nlp_adversarial_example_master_pytorch')


def data_loading(train_text, test_text, train_target, test_target):
    dataset = data_infor(train_text, train_target)
    len_train = len(dataset)
    indx = list(range(len_train))
    train_data = Subset(dataset, indx)
    
    dataset = data_infor(test_text, test_target)
    len_test = len(dataset)
    indx = list(range(len_test))
    test_data = Subset(dataset, indx)
    return train_data, test_data, len_train

def imdb_run():
    args = parser.parse_args()

    data = args.data
    freeze = args.freeze
    nlayer = args.nlayer
    kept_prob = args.kept_prob_dropout
    bert_lstm_save_path=args.save_path
    learning_rate = args.learning_rate
    if data.lower() == 'imdb':
        data_path = 'aclImdb'
        
    
    bert = BertModel.from_pretrained('bert-base-uncased')
    
    max_len = 250
    max_vocab = bert.config.to_dict()['vocab_size']-3
    data_processed = pre_processing(data_path, max_vocab)
    train_sequences, test_sequences = data_processed.seqs_num()
    train_text_init, test_text_init = data_processed.numerical(train_sequences, test_sequences, max_len = max_len)
    
    
    train_text = pad_sequences(train_text_init, maxlen = max_len, padding = 'post')
    test_text = pad_sequences(test_text_init, maxlen = max_len, padding = 'post')
    train_target = data_processed.all_train_labels
    test_target = data_processed.all_test_labels
    
    train_data, test_data, length_train = data_loading(train_text, test_text, train_target, test_target)
    
    
    
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    BatchSize = int(length_train/100)
    train_loader = DataLoader(train_data, batch_size = BatchSize, shuffle = True, pin_memory = True)
    test_loader = DataLoader(test_data, batch_size = BatchSize, shuffle = True, pin_memory = True)

    model = bert_lstm(bert, 2, False, nlayer, 128, freeze, kept_prob)
    model.to(device)
    
    
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW([cont for cont in model.parameters() if cont.requires_grad], lr = learning_rate)
    
    epoches = 200
    best_epoch = 0
    best_acc = 0
    patience = 20
    
    for epoch in range(epoches):
      test_pred = torch.tensor([])
      test_targets = torch.tensor([])
      train_pred = torch.tensor([])
      train_targets = torch.tensor([])
      test_loss = []
      train_loss = []
    
      model.train()
      for batch_index, (seqs, length, target) in enumerate(train_loader):
        seqs, target, length = seqs.to(device), target.to(device), length.to(device)
        seqs = seqs.type(torch.LongTensor)
        args = torch.argsort(length, descending = True)
        length = length[args]
        seqs = seqs[args][:, 0:length[0]]
        
        target = target[args].type(torch.LongTensor)
        optimiser.zero_grad()
        output = model(seqs, length)
        loss = criterion(output, target)
        loss.backward()
        optimiser.step()
    
        train_pred = torch.cat((train_pred, output.cpu()), dim = 0)
        train_targets = torch.cat((train_targets, target.type(torch.float).cpu()))
        train_loss.append(loss)
        
        if batch_index % 10 == 0:
          print('Train Batch:{}, Train Loss:{:.4f}.'.format(batch_index, loss.item()))
    
      train_accuracy = model.evaluate_accuracy(train_pred.detach().numpy(), train_targets.detach().numpy())
      print('Epoch:{}, Train Accuracy:{:.4f}, Train Mean loss:{:.4f}.'.format(epoch, train_accuracy, sum(train_loss)/len(train_loss)))
      print("\n")
    
      model.eval()
      with torch.no_grad():
        for batch_index, (seqs, length, target) in enumerate(test_loader):
          seqs, target, length = seqs.to(device), target.to(device), length.to(device)
          seqs = seqs.type(torch.LongTensor)
          len_order = torch.argsort(length, descending = True)
          length = length[len_order]
          seqs = seqs[len_order]
          target = target[len_order]
    
          output = model(seqs, length)
          test_pred = torch.cat((test_pred, output.cpu()), dim = 0)
          test_targets = torch.cat((test_targets, target.type(torch.float).cpu()))
          loss = criterion(output, target)
          test_loss.append(loss.item())
          if batch_index % 10 == 0:
            print('Test Batch:{}, Test Loss:{:.4f}.'.format(batch_index, loss.item()))
        accuracy = model.evaluate_accuracy(test_pred.numpy(), test_targets.numpy())
        print('Epoch:{}, Test Accuracy:{:.4f}, Test Mean loss:{:.4f}.'.format(epoch, accuracy, sum(test_loss)/len(test_loss)))
        # best save
        if accuracy > best_acc:
          best_acc = accuracy
          best_epoch = epoch
          torch.save(model.state_dict(), bert_lstm_save_path)
        # early stop
        if epoch-best_epoch >=patience:
          print('Early stopping')
          print('Best epoch: {}, Best accuracy: {:.4f}.'.format(best_epoch, best_acc))
          print('\n\n')
          break
  
    
if __name__ == '__main__':
    imdb_run()