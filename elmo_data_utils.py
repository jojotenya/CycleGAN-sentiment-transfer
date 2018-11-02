import os
import numpy as np
import json
import re
import random
from elmoformanylangs import Embedder

class data_utils():
    def __init__(self,args):
        self.data_path = 'data/feature_twitter.txt'
        self.elmo_path = '/data/ELMoForManyLangs/elmo_chinese'
        self.num_batch = 12
        self.sent_length = args.sequence_length
        self.batch_size = args.batch_size
        self.emb = Embedder(self.elmo_path,use_gpu=False)

    def pad_sentence(self,sentence,pad_tag="<eos>"):
        if self.sent_length > len(sentence):
            pad_num = self.sent_length - len(sentence) 
            sentence += [pad_tag]*pad_num
        return sentence

    def data_generator(self,class_id):
        while True:
            with open(self.data_path) as fp:
                for line in fp:
                    s = line.strip().split('+++$+++')
                    if int(s[0])==class_id and random.randint(0,10) >= 2:
                        row = s[1].strip().split()[:self.sent_length]
                        row_len = len(row)
                        row = self.pad_sentence(row)
                        yield row, row_len 

    def X_data_generator(self):
        return self.data_generator(0)

    def Y_data_generator(self):
        return self.data_generator(1)


    def gan_data_generator(self):
        one_X_batch = []
        one_Y_batch = []
        one_X_len_batch = []
        one_Y_len_batch = []

        for (one_X,one_X_len),(one_Y,one_Y_len) in zip(self.X_data_generator(),self.Y_data_generator()):
            one_X_batch.append(one_X)
            one_Y_batch.append(one_Y)
            one_X_len_batch.append(one_X_len)
            one_Y_len_batch.append(one_Y_len)
            if len(one_X_batch) == self.batch_size*self.num_batch:
                one_X_batch = self.emb.sents2elmo(one_X_batch)
                one_X_batch = np.array(one_X_batch).reshape(self.num_batch,self.batch_size,self.sent_length,-1)
                one_Y_batch = self.emb.sents2elmo(one_Y_batch)
                one_Y_batch = np.array(one_Y_batch).reshape(self.num_batch,self.batch_size,self.sent_length,-1)
                one_X_len_batch = np.array(one_X_len_batch).reshape(self.num_batch,self.batch_size,)
                one_Y_len_batch = np.array(one_Y_len_batch).reshape(self.num_batch,self.batch_size,)
                yield one_X_batch,one_X_len_batch,one_Y_batch,one_Y_len_batch
                one_X_batch = []
                one_Y_batch = []
                one_X_len_batch = []
                one_Y_len_batch = []


    def pretrain_generator_data_generator(self):
        one_X_batch = []
        one_Y_batch = []
        one_X_len_batch = []
        one_Y_len_batch = []

        for (one_X,one_X_len),(one_Y,one_Y_len) in zip(self.X_data_generator(),self.Y_data_generator()):
            one_X_batch.append(one_X)
            one_Y_batch.append(one_Y)
            one_X_len_batch.append(one_X_len)
            one_Y_len_batch.append(one_Y_len)
            if len(one_X_batch) == self.batch_size:
                one_X_batch = self.emb.sents2elmo(one_X_batch)
                one_X_batch = np.array(one_X_batch)
                one_Y_batch = self.emb.sents2elmo(one_Y_batch)
                one_Y_batch = np.array(one_Y_batch)
                one_X_len_batch = np.array(one_X_len_batch)
                one_Y_len_batch = np.array(one_Y_len_batch)
                yield one_X_batch,one_X_len_batch,one_Y_batch,one_Y_len_batch
                one_X_batch = []
                one_Y_batch = []
                one_X_len_batch = []
                one_Y_len_batch = []


    def test_data_generator(self):
        one_batch = np.zeros([self.batch_size,self.sent_length])
        batch_count = 0
        for line in open('seq2seq_test.txt'):
            one_batch[batch_count] = self.sent2id(line.strip())
            batch_count += 1
            if batch_count == self.batch_size:
                yield one_batch
                batch_count = 0
                one_batch = np.zeros([self.batch_size,self.sent_length])

        if batch_count >= 1:
            yield one_batch
