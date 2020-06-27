# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:54:12 2020

@author: 13758
"""
import os
import re
import numpy as np
from collections import OrderedDict, defaultdict


class pre_processing(object):
  def __init__(self, path, max_vocab):
    # extract all dataset
    self.max_vocab = max_vocab
    self.train_path_pos = os.path.join(path, 'train/pos')
    self.train_path_neg = os.path.join(path, 'train/neg')
    self.test_path_pos = os.path.join(path, 'test/pos')
    self.test_path_neg = os.path.join(path, 'test/neg')

    train_pos_files = [os.path.join(self.train_path_pos, i) for i in os.listdir(self.train_path_pos) if i.endswith('.txt')]
    train_neg_files = [os.path.join(self.train_path_neg, i) for i in os.listdir(self.train_path_neg) if i.endswith('.txt')]
    test_pos_files = [os.path.join(self.test_path_pos, i) for i in os.listdir(self.test_path_pos) if i.endswith('.txt')]
    test_neg_files = [os.path.join(self.test_path_neg, i) for i in os.listdir(self.test_path_neg) if i.endswith('.txt')]
    
    self.train_pos_seqs = [open(i).read().lower() for i in train_pos_files]
    self.test_pos_seqs = [open(i).read().lower() for i in test_pos_files]
    self.train_neg_seqs = [open(i).read().lower() for i in train_neg_files]
    self.test_neg_seqs = [open(i).read().lower() for i in test_neg_files]

    
  def processing(self):
    
    self.train_pos_tokens = [re.sub(r"[^A-Za-z0-9_']", ' ', seq) for seq in self.train_pos_seqs]
    self.train_pos_labels = np.ones((1, len(self.train_pos_tokens)))
    self.train_neg_tokens = [re.sub(r"[^A-Za-z0-9_']", ' ', seq) for seq in self.train_neg_seqs]
    self.train_neg_labels = np.zeros((1, len(self.train_neg_tokens)))

    self.all_train = self.train_pos_tokens + self.train_neg_tokens
    self.all_train = [[token.strip("'") for token in seq.strip().split()] for seq in self.all_train]

    self.all_train_labels = np.concatenate((self.train_pos_labels, self.train_neg_labels),1)

    self.words_indx = self.token_indx()

    self.indx_word = {}
    for word in self.words_indx:
      self.indx_word[self.words_indx[word]] = word
    

    self.test_pos_tokens = [re.sub(r"[^A-Za-z0-9_']", ' ', seq) for seq in self.test_pos_seqs]
    self.test_pos_labels = np.ones((1, len(self.test_pos_tokens)))
    self.test_neg_tokens = [re.sub(r"[^A-Za-z0-9_']", ' ', seq) for seq in self.test_neg_seqs]
    self.test_neg_labels = np.zeros((1, len(self.test_neg_tokens)))
    self.all_test_labels = np.concatenate((self.test_pos_labels,self.test_neg_labels), 1)
    self.all_test = self.test_pos_tokens+self.test_neg_tokens
    self.all_test = [[token.strip("'") for token in seq.strip().split()] for seq in self.all_test]

  def token_indx(self):
    # length = 1
    short = OrderedDict()
    short = OrderedDict()
    # normal words
    normal = OrderedDict()
    normal = OrderedDict()
    doc = defaultdict(int)
    # stop words/appear many times
    for tokens in self.all_train:
      # tokens = seq.strip().split()
      # tokens = [token.strip("'") for token in tokens]
      for token in tokens:
        # token = token.strip("'")
        if len(token) <= 1:
          if token in short:
            short[token] += 1
          else: 
            short[token] = 1
        else:
          if token in normal:
            normal[token] += 1
          else:
            normal[token] = 1
      for token in set(tokens):
        doc[token] += 1

    if len(normal) >= self.max_vocab:
      normal_list = self.order(normal, doc)
      short_list = self.order(short, doc)
      list_ = normal_list+short_list
      list_.insert(99, 'UNK')
      # list_.insert(100, 'CLS')
      # list_.insert(101, 'SEP')
      word_indx = dict(zip(list_, list(range(1, len(list_)+1))))
    else:
      list_ = self.order(normal+short, doc)
      list_.insert(99, 'UNK')
      # list_.insert(100, 'CLS')
      # list_.insert(101, 'SEP')
      
      word_indx = dict(zip(list_, list(range(1, len(list_)+1))))
    return word_indx
  
  def order(self, dictionary, doc):
    indx_num = OrderedDict()
    # print(normal['hh'])
    for token in dictionary:
      indx_num[token] = np.log(doc[token]*dictionary[token])
    indx_num = list(indx_num.items())
    indx_num.sort(key = lambda x: x[1], reverse = True)
    word_list = [token[0] for token in indx_num]
    return word_list


  def seqs_num(self):
    self.train_seqs = [[self.words_indx[token] if self.words_indx[token]<self.max_vocab else self.words_indx['UNK'] for token in seq] for seq in self.all_train]
    self.test_seqs = []
    for seq in self.all_test:
      tokens = []
      for token in seq:
        if (token in self.words_indx) and (self.words_indx[token]<self.max_vocab):
          tokens.append(self.words_indx[token])
        else:
          tokens.append(self.words_indx['UNK'])
      self.test_seqs.append(tokens)
    # self.test_seqs = [[self.words_indx[token] if (token in self.words_indx) and (self.words_indx[token]<self.max_vocab) else self.max_vocab for token in seq] for seq in self.all_test]
    return self.train_seqs, self.test_seqs

  def bert_tokenize(self, tokenizer):
    self.train_pos_tokens = [tokenizer.tokenize(seq) for seq in self.train_pos_seqs]
    self.train_pos_labels = np.ones((1, len(self.train_pos_tokens)))
    self.train_neg_tokens = [tokenizer.tokenize(seq) for seq in self.train_neg_seqs]
    self.train_neg_labels = np.zeros((1, len(self.train_neg_tokens)))

    self.all_train = self.train_pos_tokens + self.train_neg_tokens
    self.all_train_labels = np.concatenate((self.train_pos_labels, self.train_neg_labels),1)

    self.test_pos_tokens = [tokenizer.tokenize(seq) for seq in self.test_pos_seqs]
    self.test_pos_labels = np.ones((1, len(self.test_pos_tokens)))
    self.test_neg_tokens = [tokenizer.tokenize(seq) for seq in self.test_neg_seqs]
    self.test_neg_labels = np.zeros((1, len(self.test_neg_tokens)))

    self.all_test_labels = np.concatenate((self.test_pos_labels,self.test_neg_labels), 1)
    self.all_test = self.test_pos_tokens+self.test_neg_tokens

  def numerical(self, tokenizer, train_seqs, test_seqs, max_len):

    train_text = [[tokenizer.cls_token_id]+ seq + [tokenizer.sep_token_id] if len(seq)<=max_len-2 else [tokenizer.cls_token_id]+ seq[0:max_len-2] + [tokenizer.sep_token_id] for seq in train_seqs]
    test_text = [[tokenizer.cls_token_id]+ seq + [tokenizer.sep_token_id] if len(seq)<=max_len-2 else [tokenizer.cls_token_id]+ seq[0:max_len-2] + [tokenizer.sep_token_id] for seq in test_seqs]
    return train_text, test_text
  
  def bert_indx(self, tokenizer):
    train_seqs = [tokenizer.convert_tokens_to_ids(seq) for seq in self.all_train]
    test_seqs = [tokenizer.convert_tokens_to_ids(seq) for seq in self.all_test]
    return train_seqs, test_seqs