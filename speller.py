import os
import random
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from torch.autograd import Variable
import csv

from torch.utils.data import Dataset, DataLoader
import time 
#from phoneme_list import *
from ctcdecode import CTCBeamDecoder
from torch.nn import CTCLoss
import Levenshtein as Lev


letter_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',\
             'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<eos>']

letter_to_index= {'<eos>': 0, ' ': 1, "'": 2, '+': 3, '-': 4, '.': 5, 'A': 6, 'B': 7, 'C': 8, 'D': 9, 'E': 10, 'F': 11, \
                'G': 12, 'H': 13, 'I': 14, 'J': 15, 'K': 16, 'L': 17, 'M': 18, 'N': 19, 'O': 20, 'P': 21, 'Q': 22, 'R': 23,\
                 'S': 24, 'T': 25, 'U': 26, 'V': 27, 'W': 28, 'X': 29, 'Y': 30, 'Z': 31, '_': 32}

# Speller specified in the paper
class Speller(nn.Module):
    def __init__(self, n_classes, speller_hidden_dim, speller_rnn_layer, attention, context_dim):
        super(Speller, self).__init__()
        self.n_classes = n_classes
        self.rnn_unit = nn.LSTMCell

        # self.rnn_layer = self.rnn_unit(output_class_dim + speller_hidden_dim, speller_hidden_dim,
        #                                num_layers=speller_rnn_layer)

        self.rnn_layer = torch.nn.ModuleList()
        self.rnn_inith = torch.nn.ParameterList()
        self.rnn_initc = torch.nn.ParameterList()

        self.rnn_layer.append(nn.LSTMCell(speller_hidden_dim + context_dim, hidden_size=speller_hidden_dim))
        self.rnn_initc.append(nn.Parameter(torch.rand(1, speller_hidden_dim)))
        self.rnn_inith.append(nn.Parameter(torch.rand(1, speller_hidden_dim)))

        for i in range(speller_rnn_layer):
            if i != 0:
                self.rnn_layer.append(nn.LSTMCell(speller_hidden_dim, speller_hidden_dim))
                self.rnn_inith.append(nn.Parameter(torch.rand(1, speller_hidden_dim)))
                self.rnn_initc.append(nn.Parameter(torch.rand(1, speller_hidden_dim)))

        self.attention = attention

        # char embedding
        self.embed = nn.Embedding(n_classes, speller_hidden_dim)
       
        # prob output layers
        self.fc = nn.Linear(speller_hidden_dim + context_dim, speller_hidden_dim)
        self.activate = torch.nn.LeakyReLU(negative_slope=0.2)
        self.unembed = nn.Linear(speller_hidden_dim, n_classes)
        
        self.unembed.weight = self.embed.weight #Tying the weights of embedding and the scoring layer
        self.character_distribution = nn.Sequential(self.fc, self.activate, self.unembed)
        
    def forward(self, listener_feature, seq_sizes, max_iters, ground_truth=None, teacher_force_rate=0.9, dropout=[]):
       
        if ground_truth is None:
            teacher_force_rate = 0
        teacher_force = True if np.random.random_sample() < teacher_force_rate else False
       
        batch_size = listener_feature.shape[0]
        
        state, output_word = self.get_initial_state(batch_size)
             
        # dropouts
        dropout_masks = []
        if dropout and self.training:
            h = state[0][0] # B, C
            n_layers = len(state[0])

            for i in range(n_layers):
                
                mask = h.data.new(h.size(0), h.size(1)).bernoulli_(1 - dropout[i]) / (1 - dropout[i])
                dropout_masks.append(Variable(mask, requires_grad=False))

        raw_pred_seq = []
        attention_record = []

        for step in range(ground_truth.size(1) if ground_truth is not None else max_iters):
            ''' Iterating from 0 to the max_len'''
            # print("last_output_word_forward", idx2chr[output_word.data[0]])
            attention_score, raw_pred, state = self.run_one_step(listener_feature, seq_sizes, output_word, state, dropout_masks=dropout_masks)
            
            
            attention_record.append(attention_score)
            raw_pred_seq.append(raw_pred)

            # Teacher force - use ground truth as next step's input
            if teacher_force:
                output_word = ground_truth[:, step]
            else:

                output_word = torch.max(raw_pred, dim=1)[1]
                      
        return torch.stack(raw_pred_seq, dim=1), attention_record

    def run_one_step(self, listener_feature, seq_sizes, last_output_word, state, dropout_masks=None):
        
        output_word_emb = self.embed(last_output_word)
        
        # get attention context
        hidden, cell = state[0], state[1]
        last_rnn_output = hidden[-1]  # last layer

        # print("last_rnn_output", last_rnn_output)
        # print("listener_feature", listener_feature)
        
        attention_score, context = self.attention(last_rnn_output, listener_feature, seq_sizes)
        # Context = batch x context_dim : 128  
        # run speller rnns for one time step
        
        rnn_input = torch.cat([output_word_emb, context], dim=1)

        new_hidden, new_cell = [None] * len(self.rnn_layer), [None] * len(self.rnn_layer)
        
        # If number of layers are 3 [none,none,none]
        for l, rnn in enumerate(self.rnn_layer):

            new_hidden[l], new_cell[l] = rnn(rnn_input, (hidden[l],cell[l]))
            if dropout_masks:
                rnn_input = new_hidden[l] * dropout_masks[l]
            else:
                rnn_input = new_hidden[l]

        rnn_output = new_hidden[-1]  # last layer

        # make prediction
        concat_feature = torch.cat([rnn_output, context], dim=1)

        #             print("concat_feature.size()", concat_feature.size())
        raw_pred = self.character_distribution(concat_feature)
        
        #             print("raw_pred.size()", raw_pred.size())

    
        return attention_score, raw_pred, [new_hidden, new_cell]

    def get_initial_state(self, batch_size=32):
        
        hidden=[]
        for h in self.rnn_inith:
            hidden.append(h.repeat(batch_size,1))
            
        cell=[]
        for c in self.rnn_initc:
            cell.append(c.repeat(batch_size,1))

        #assert len(hidden) == len(cell)
        
        #input ('')
        #------hideen = batch * 256
        # <sos> (same vocab as <eos>)
        output_word = Variable(hidden[0].data.new(batch_size).long().fill_(letter_to_index['<eos>']))
        # 3 * 256 , 3 * 256
        return [hidden, cell], output_word


class mydecoder (nn.Module):

    def __init__(self,vocab_size,decoder_hidden_dim,n_layers,attention, context_dim):
        super(Decoder,self).__init__()

        self.vocab_size= vocab_size

        self.lstm1=nn.LSTMCell(decoder_hidden_dim+context_dim,decoder_hidden_dim)
        self.lstm2=nn.LSTMCell(decoder_hidden_dim,decoder_hidden_dim)
        self.lstm3=nn.LSTMCell(decoder_hidden_dim,decoder_hidden_dim)

        self.attention=attention

        self.embed= nn.Embedding(vocab_size,decoder_hidden_dim)

        self.fc= nn.Linear(decoder_hidden_dim+context_dim,decoder_hidden_dim)
        self.activate= nn.ReLU()
        self.scoring= nn.Linear(decoder_hidden_dim, vocab_size)

        self.scoring.weight = self.embed.weight

        self.out_layere=nn.Sequential(self.fc,self.activate,self.scoring)

    def forward(self, listener_feature, seq_sizes, max_iters, ground_truth=None, teacher_force_rate=0.9, dropout=[]):

        batch_size=listener_feature.shape[0]

        state,output = self.get_initial_state(batch_size)


class Encoder(nn.Module):
    def __init__(self, input_dimension, hidden_dimension,value_size=128,key_size=128,dropout=0):

        super(Listener_Block,self).__init__()
        #self.rnn_unit=getattr(nn,'LSTM'.upper())
        self.n_layers=3
            
        self.lstm_layer_1=nn.LSTM(input_dimension * 2,hidden_dimension,1,bidirectional=True,dropout=dropout,batch_first=True)
        self.lstm_layer_2=nn.LSTM(hidden_dimension*2 * 2,hidden_dimension,1,bidirectional=True,dropout=dropout,batch_first=True)
        self.lstm_layer_3=nn.LSTM(hidden_dimension*2 * 2,hidden_dimension,1,bidirectional=True,dropout=dropout,batch_first=True)        
        
        
        '''
        self.lstm_layer_1=pBLSTMLayer(input_dimension=input_dimension,hidden_dimension=hidden_dimension,dropout=dropout)
        self.lstm_layer_2=pBLSTMLayer(input_dimension=hidden_dimension*2,hidden_dimension=hidden_dimension,dropout=dropout)
        self.lstm_layer_3=pBLSTMLayer(input_dimension=hidden_dimension*2,hidden_dimension=hidden_dimension,dropout=dropout)
        self.n_layers=3
        '''
        self.layers=[self.lstm_layer_1,self.lstm_layer_2,self.lstm_layer_3]
        print (self.lstm_layer_1,self.lstm_layer_2,self.lstm_layer_3)

        self.key_network = nn.Linear(hidden_dimension*2, value_size)
        self.value_network = nn.Linear(hidden_dimension*2, key_size)
  
        
    def forward(self,frames,sequence_length):
        #print ("Input shape",frames.shape)
        input_data= frames.permute(1,0,2).contiguous()
        x=input_data.to(device)
        
        for i in range (self.n_layers):
            batch, seq_len , dimensions=input_data.shape
            
            if seq_len % 2 == 0:
                pass
            else:
                seq_len=seq_len-1
                input_data=input_data[ :, :-1 ,:]

            input_data=input_data.contiguous().view(batch, seq_len//2 , dimensions *2)
            
            output, hidden= self.layers[i](input_data)
            #print (output.shape)
            input_data=output

        keys=self.key_network(output)
        value=self.value_network(output)

        print (keys.shape,values.shape)
        input ('')
        out_seq_len= [size // (2*2*2) for size in sequence_length]
        #print (output.shape,out_seq_len)
        return output , out_seq_len

def SequenceWise(input_module, input_x):
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    reshaped_x = input_x.contiguous().view(-1, input_x.size(-1))
    output_x = input_module(reshaped_x)
    return output_x.view(batch_size, time_steps, -1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size,context_size=128):
        super(Speller1, self).__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm1 = nn.LSTMCell(hidden_size+context_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size,key_size)

        self.attention = Attention()
        self.character_prob = nn.Linear(key_size+context_size, vocab_size)

    def forward(self, x, context, context_lengths, state=None):
        """
        :param x: (N,), target tokens in the current timestep
        :param context: (N, T, H), encoded source sequences
        :param context_lengths: (N,) lengths of source sequences
        :param state: LSTM hidden states from the last timestep (or from the encoder for the first step)
        :returns: prediction of target tokens in the next timestep, LSTM hidden states of the current timestep, and attention vectors
        """
        x = self.embed(x)
        new_state = self.lstm(x, state)
        x = new_state[0]
        x_att, attention = self.attention(x, context, context_lengths)
        x = torch.cat([x, x_att], dim=1)
        return self.output(x), new_state, attention

