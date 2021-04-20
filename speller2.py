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
#from ctcdecode import CTCBeamDecoder
from torch.nn import CTCLoss
import Levenshtein as Lev
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


letter_list = ['<eos>', ' ', "'", '+', '-', '.','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',\
             'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','_','<sos>']

letter_to_index= {'<eos>': 0, ' ': 1, "'": 2, '+': 3, '-': 4, '.': 5, 'A': 6, 'B': 7, 'C': 8, 'D': 9, 'E': 10, 'F': 11, \
                'G': 12, 'H': 13, 'I': 14, 'J': 15, 'K': 16, 'L': 17, 'M': 18, 'N': 19, 'O': 20, 'P': 21, 'Q': 22, 'R': 23,\
                 'S': 24, 'T': 25, 'U': 26, 'V': 27, 'W': 28, 'X': 29, 'Y': 30, 'Z': 31, '_': 32,'<sos>':33}

class Encoder(nn.Module):
    def __init__(self, input_dimension, hidden_dimension,value_size=128,key_size=128,dropout=0):

        super(Encoder,self).__init__()
        #self.rnn_unit=getattr(nn,'LSTM'.upper())
        self.n_layers=3
            
        self.lstm_layer_1=nn.LSTM(input_dimension ,hidden_dimension,1,bidirectional=True,batch_first=True)
        self.lstm_layer_2=nn.LSTM(hidden_dimension*2 ,hidden_dimension,1,bidirectional=True,batch_first=True)
        self.lstm_layer_3=nn.LSTM(hidden_dimension*2 ,hidden_dimension,1,bidirectional=True,batch_first=True)
        self.lstm_layer_4=nn.LSTM(hidden_dimension*2, hidden_dimension,1,bidirectional=True,batch_first=True)        
        
        
        
        self.layers=[self.lstm_layer_2,self.lstm_layer_3,self.lstm_layer_4]
        
        self.key_network=nn.Linear(hidden_dimension *2 , value_size)
        nn.init.xavier_normal_(self.key_network.weight)
        self.value_network=nn.Linear(hidden_dimension * 2, key_size)
        nn.init.xavier_normal_(self.value_network.weight)


        print (self.key_network)
        print (self.value_network)
        #self.key_network = nn.Linear(hidden_dimension*2, key_size)
        #self.value_network = nn.Linear(hidden_dimension*2, value_size)
  
        
    def forward(self,frames,sequence_length):
        #print ("Input shape",frames.shape)
        input_data= frames.permute(1,0,2).contiguous()
       
        input_data = nn.utils.rnn.pack_padded_sequence(input_data, lengths=sequence_length, batch_first=True, enforce_sorted=False)
        output,_ = self.lstm_layer_1(input_data)
        
        
        for i in range (self.n_layers):
            #batch, seq_len , dimensions=output_data.shape
            output,_ =self.layers[i](output)
            unpacked_tensor,unpack_len=nn.utils.rnn.pad_packed_sequence(output,batch_first=True)
            
            if unpacked_tensor.shape[1] % 2 == 0:
                pass
            else:
                 ##Something to make it even otherwise error comes in the next step
                unpacked_tensor=unpacked_tensor[ :, :-1 ,:]

            batch_size,seq_len,dimensions= unpacked_tensor.shape
            
            unpacked_tensor=unpacked_tensor.unsqueeze(2).reshape(batch_size, seq_len//2 ,2, dimensions)

            output= unpacked_tensor.mean(dim=2)
            
            output=nn.utils.rnn.pack_padded_sequence(output,lengths=unpack_len//2,batch_first=True,enforce_sorted=False)
            
            
        #keys=self.key_network(output)
        #values=self.value_network(output)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = output.transpose(1,0)
        keys  = self.key_network(output)
        values= self.value_network(output)
        
        out_seq_len= [size // (2*2*2) for size in sequence_length]
        #print (output.shape,out_seq_len)
        
        return keys,values, out_seq_len


class Attention (nn.Module):
    def __init__(self):
        super(Attention,self).__init__()

    def forward (self, query, key, value, lengths):
        lengths=torch.Tensor(lengths).long()
        "Key is encoder output "
        "Query is decoder output"
        "Value is context"

        #attention = torch.bmm(context, query.unsqueeze(2)).squeeze(2)
        energy = torch.bmm (key.transpose(1,0), query.unsqueeze(2)).squeeze(2)

        mask = torch.arange(key.size(0)).unsqueeze(0) >= lengths.unsqueeze(1)

        energy.masked_fill_(mask.to(device), -1e9)
        attention = nn.functional.softmax(energy, dim=1)
        
        out = torch.bmm(attention.unsqueeze(1), value.transpose(1,0)).squeeze(1)
        #print (attention.shape, out.shape)
        #input ('')
        " out is context"
        return attention, out 

        
        
        return attention_score, context
def SequenceWise(input_module, input_x):
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    reshaped_x = input_x.contiguous().view(-1, input_x.size(-1))
    output_x = input_module(reshaped_x)
    return output_x.view(batch_size, time_steps, -1)
class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size,value_size=128, key_size=128,context_size=128,isAttended=False): #Inititally key size was 128
        super(Decoder, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, 256)
        self.lstm1 = nn.LSTMCell(256+ value_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size,hidden_size)
        self.lstm3 = nn.LSTMCell(hidden_size,key_size)

        #self.attention = Attention_Block(key_query_dim=128, speller_query_dim=hidden_size, listener_feature=512, context_dim=128)
        self.isAttended=isAttended
        if self.isAttended:
            self.attention= Attention()

        
        
        " Heree value size is same as the context size"
        self.linear_out= nn.Linear ( key_size + value_size, key_size)
        self.dropout= nn.Dropout (0.3)
        self.character_prob = nn.Linear(key_size, vocab_size)
       
    def forward(self, keys, values, seq_lengths, text=None, train=True, teacher_force_rate=0.9):
        
        " Remember key = listener_feature, context = seq_lengths   , text=labels"
        """
        :param x: (N,), target tokens in the current timestep
        :param context: (N, T, H), encoded source sequences
        :param context_lengths: (N,) lengths of source sequences
        :param state: LSTM hidden states from the last timestep (or from the encoder for the first step)
        :returns: prediction of target tokens in the next timestep, LSTM hidden states of the current timestep, and attention vectors
        """
        '''
        :param key :(T,N,key_size) Output of the Encoder Key projection layer
        :param values: (T,N,value_size) Output of the Encoder Value projection layer
        :param text: (N,text_len) Batch input of text with text_length
        :param train: Train or eval mode
        :return predictions: Returns the character perdiction probability 
        '''
        #print ("Input to decoder ",encoder_output.shape)
        


        #print ("Length of text",text.shape)
        
        batch_size=keys.shape[1]
        
        if text is None: #Testing mode if text is not present
            teacher_force_rate = 0
        teacher_force = True if np.random.random_sample() < teacher_force_rate else False
        
        #output_word,state=self.get_initial_state(batch_size)
       
        if train or text is not None:
            max_len= text.shape[1]
            embeddings= self.embed(text)
             
        else:
            max_len= 250
    

        #new_hidden_states = [None, None]
        prediction_list=[]
        attention_list=[]
        prediction= torch.zeros(batch_size,1).to(device) #First input 
        hidden_states=[None,None,None]
        
        
        for i in range (max_len):
            if teacher_force and train :
                char_embed= embeddings[:,i,:]
                "feeding the actual word is the character embeddings"
            else:
                char_embed= self.embed(prediction.argmax(dim=1))    
                #char_embed= self.embed(torch.max(prediction,dim=1)[1])


            if i > 0:
                inp_1  = torch.cat ([char_embed,context],dim=1)
            else:
                inp_1  = torch.cat ([char_embed,values[-1,:,:]],dim=1)


            #out_word_embedding= self.embed (output_word) ##Getting the embedding 
            hidden_states[0]= self.lstm1(inp_1,hidden_states[0])

            inp_2= hidden_states[0][0]
        
            hidden_states[1]= self.lstm2(inp_2,hidden_states[1])

            inp_3 = hidden_states[1][0]

            hidden_states[2] = self.lstm3(inp_3, hidden_states[2])

            output=hidden_states[2][0]
            #print (output.shape)
            attention, context= self.attention(output, keys, values, seq_lengths)
            
            prediction = self.linear_out ( torch. cat([  output, context], dim=1))
            prediction = self.dropout(prediction)
            prediction = self.character_prob(prediction)
            prediction_list.append(prediction.unsqueeze(1))
            attention_list.append(attention)
            #state=[new_hidden,new_cell]
            
       
        return torch.cat(prediction_list,dim=1) , attention_list
    
class Seq2Seq(nn.Module):
  def __init__(self,input_dim,vocab_size,encoder_hidden_dim,decoder_hidden_dim):# value_size=128, key_size=128,isAttended=False):
    super(Seq2Seq,self).__init__()
    'hidden_dim is passed from main and is currently 256'

    self.encoder = Encoder(input_dim, encoder_hidden_dim) #Encoder will output of 512 dimensions if we feed 256 dimensions as hidden
    self.decoder = Decoder(vocab_size, decoder_hidden_dim,isAttended=True) #Here 256 is the decoder hidden dimensions not the encoder.

  def forward(self,speech_input, speech_len, text_input=None,train=True,teacher_force=0.9,max_len=250):
    key, value, seq_lengths = self.encoder(speech_input, speech_len)

    if train =='train':
      predictions,attention = self.decoder(key,value,seq_lengths,text_input,train=True,teacher_force_rate=teacher_force)
     
    elif train =='valid':
      predictions,attention = self.decoder(key,value,seq_lengths,text_input,train=False)
     
    elif train == 'test':
      predictions,attention = self.decoder(key,value,seq_lengths, text=None, train=False)
    
   
    return predictions,attention


