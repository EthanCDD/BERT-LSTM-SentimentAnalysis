# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:44:44 2020

@author: 13758
"""
import numpy as np
from torch.utils.data import Dataset

class data_infor(Dataset):
  def __init__(self, seqs, target):
    self.seqs = seqs
    self.target = target
   
  def __getitem__(self, index):
    seq = self.seqs[index]
    l = np.sum(np.sign(self.seqs[index]))
    target = self.target[0,index]
    return seq, l, target

  def __len__(self):
    return len(self.seqs)
