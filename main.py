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

from speller import *

idx2chr = ['<eos>', ' ', "'", '+', '-', '.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_']
chr2idx = {'<eos>': 0, ' ': 1, "'": 2, '+': 3, '-': 4, '.': 5, 'A': 6, 'B': 7, 'C': 8, 'D': 9, 'E': 10, 'F': 11, 'G': 12, 'H': 13, 'I': 14, 'J': 15, 'K': 16, 'L': 17, 'M': 18, 'N': 19, 'O': 20, 'P': 21, 'Q': 22, 'R': 23, 'S': 24, 'T': 25, 'U': 26, 'V': 27, 'W': 28, 'X': 29, 'Y': 30, 'Z': 31, '_': 32}

letter_list = ['<eos>', ' ', "'", '+', '-', '.','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',\
             'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','_']

letter_to_index= {'<eos>': 0, ' ': 1, "'": 2, '+': 3, '-': 4, '.': 5, 'A': 6, 'B': 7, 'C': 8, 'D': 9, 'E': 10, 'F': 11, \
                'G': 12, 'H': 13, 'I': 14, 'J': 15, 'K': 16, 'L': 17, 'M': 18, 'N': 19, 'O': 20, 'P': 21, 'Q': 22, 'R': 23,\
                 'S': 24, 'T': 25, 'U': 26, 'V': 27, 'W': 28, 'X': 29, 'Y': 30, 'Z': 31, '_': 32}

assert len(letter_list) == len(letter_to_index)

def transform_letter_to_index(transcript):

    output_transcript=[]
   
    for idx, list_of_words in enumerate(transcript): #list_of_wowrds = ['THE' 'INK' 'IS' 'ON' 'THE' 'WALL']
        
        char_sequence=[]
        for idx,words in enumerate(list_of_words): #words = THE
            char_sequence += words+" "
            
        char_sequence =char_sequence[:-1] +['<eos>']

        char_indexes=np.array([letter_to_index[i] for i in char_sequence])

        output_transcript.append(char_indexes)
        
    return np.array(output_transcript)

def convert_bstring_to_normal(inputlist):


    for idx,elements in enumerate(inputlist):
        
        elements=np.array([ele.decode('utf-8') for ele in elements])
        #print (idx,type(elements),elements)
        
        inputlist[idx]= elements
       
    return inputlist

class Language_Dataset(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        if y is not None:
            self.total_labels = sum(len(labels) for labels in y)
        else:
            self.total_labels = -1

        #print("n_utters", self.x.shape[0], "total_labels", self.total_labels)

    def __getitem__(self, idx):
        frames = self.x[idx]

        
        if self.y is None:
            labels = [-1]
        else:
            labels=self.y[idx]
            
        return torch.from_numpy(frames).float(), torch.from_numpy(np.array(labels)).int()
               

    def __len__(self):
        return self.x.shape[0]  

def collate_function(batch):
    batch_size = len(batch)
    
    batch = sorted(batch, key=lambda b: b[0].size(0), reverse=True)  # sort the batch by seq_len desc
    
    

    max_seq_len = batch[0][0].shape[0]    
    
    dimensions = batch[0][0].shape[1]
    

    pad_matrix = torch.zeros(max_seq_len, batch_size, dimensions) #----Sequence Length, Batch size , Vectors

    
    labels=torch.zeros(batch_size,max_seq_len)

    max_length=max(l.size(0) for (i,l) in batch)
    all_labels= torch.zeros(batch_size,max_length)
    
    length_of_sequence = []
    label_sizes = torch.zeros(batch_size).int()
    

    for index, (frames, labels) in enumerate(batch):
        number_of_frames = frames.shape[0]
        length_of_sequence.append(number_of_frames)

        labele_size = labels.shape[0]
        label_sizes[index] = labele_size

        pad_matrix[:number_of_frames, index, :] = frames
        all_labels[index, :labele_size] = labels  #Padding with labels only till the labele_size , the rest will be 0



    return pad_matrix, length_of_sequence, all_labels.long(), label_sizes
'''
class pBLSTMLayer (nn.Module):

    def __init__(self, input_dimension, hidden_dimension,dropout=0):
        super(pBLSTMLayer,self).__init__()
        #self.rnn_unit= getattr(nn,'LSTM'.upper())
        #print (self.rnn_unit)
        #input ('')
        self.bi_lstm=nn.LSTM(input_size=input_dimension*2 , hidden_size=hidden_dimension,num_layers= 1, bidirectional=True,dropout=dropout, batch_first=True)
        
    def forward(self,input_data):
        batch=input_data.size(0)
        timestep=input_data.size(1)

        if timestep % 2 == 0:
            pass
        else:
            input_data=input_data[:, :-1, :]
            timestep=timestep-1

        print (batch,"Timestep",timestep)
        input ('')
        feature_dim=input_data.size(2)
        print (feature_dim)
        input ('')
        input_data=input_data.contiguous().view(batch_size,timestep//2,feature_dim*2)
        print (input_data.shape)
        input ('')
        output,hidden=self.bi_lstm(input_data)

        return output,hidden
'''     


class Listener_Block(nn.Module):
    def __init__(self, input_dimension, hidden_dimension,dropout=0):

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

        '''
        print ("Input for LSTM shape",x.shape)
        input ('')
        output,hidden=self.lstm_layer_1(x)
        print ("After first layer",output.shape)
        output,hidden=self.lstm_layer_2(output)

        print ("After second layer",output.shape)
        output,hidden=self.lstm_layer_3(output)

        print ("After third layer",output.shape)

        print ("Final",output.shape)
        input ('')

        '''
        out_seq_len= [size // (2*2*2) for size in sequence_length]
        #print (output.shape,out_seq_len)
        return output , out_seq_len

def SequenceWise(input_module, input_x):
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    reshaped_x = input_x.contiguous().view(-1, input_x.size(-1))
    output_x = input_module(reshaped_x)
    return output_x.view(batch_size, time_steps, -1)

class Attention_Block(nn.Module):

    def __init__(self,key_query_dim=128, speller_query_dim=256, listener_feature=512, context_dim=128):
        super(Attention_Block,self).__init__()
        self.softmax=nn.Softmax()
        self.fc_query= nn.Linear(speller_query_dim,key_query_dim)
        self.fc_key= nn.Linear(listener_feature,key_query_dim)
        self.fc_value=nn.Linear(listener_feature,context_dim)

        self.activation = torch.nn.ReLU()
        #print (self.fc_query,self.fc_key,self.fc_value)
        #input ('')
    def forward(self, decoder_state, listener_feature, seq_sizes):
        '''
        batch_size=queries.size(0)

        hidden_size=queries.size(2)

        input_lengths=values.size(1)
        '''
        #print ("Decoder",decoder_state.shape)
        #print ("Listener",listener_feature.shape)
        #print ("Decoder State=",decoder_state.shape)
        #print ("Listener Feature=",listener_feature.shape)
       
        query= self.activation(self.fc_query(decoder_state))
        sequence_data=SequenceWise(self.fc_key,listener_feature)
        key=  self.activation(sequence_data)
        
        
        energy = torch.bmm(query.unsqueeze(1), key.transpose(1, 2)).squeeze(dim=1)
        mask = Variable(energy.data.new(energy.size(0), energy.size(1)).zero_(), requires_grad=False)

        for i, size in enumerate(seq_sizes):
            mask[i,:size]=1
        
        attention_score = self.softmax(energy)
        attention_score = mask * attention_score
        attention_score = attention_score / torch.sum(attention_score, dim=1).unsqueeze(1).expand_as(attention_score)
        
        value = self.activation(self.fc_value(listener_feature))
        context = torch.bmm(attention_score.unsqueeze(1), value).squeeze(dim=1)
        
        '''
        attn_score=torch.bmm(queries,values.transpose(1,2))

        attn_distrib=F.softmax(attn_score.view(-1,input_lengths),dim=1).view(batch_size,-1,input_lengths)

        attn_output=torch.bmm(attn_distrib,values)
        '''
        
        '''Here context is the main attention source output '''
        ''' Attention_Score=10 x 143  , Context=10 x 128'''
        
        
        return attention_score, context

class Attention (nn.Module):
    def __init__(self):
        super(Attention,self).__init__()

    def forward (self, query, context, lengths):
        lengths=torch.Tensor(lengths)
        

        attention = torch.bmm(context, query.unsqueeze(2)).squeeze(2)
        
        mask = torch.arange(context.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)
        attention.masked_fill_(mask.to(device), -1e9)
        attention = nn.functional.softmax(attention, dim=1)
        
        out = torch.bmm(attention.unsqueeze(1), context).squeeze(1)
        
        " out is context"
        return attention, out 

'''
def find_first_eos_in_pred(pred):
    # pred: L, C
    chrs = pred.max(1)[1].data.cpu().numpy()
    # print("chrs", chrs)
    for idx, c in enumerate(chrs):
        if c == letter_to_index['<eos>']:
            return idx
    return len(chrs)
'''
class Custom_Loss(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, preds, label_sizes, labels):
        # preds: B, L, C
        pred_list = []
        label_list = []
        max_iter = preds.size(1)
        


        for (pred, label, num_iter) in zip(preds, labels, label_sizes):
            pred_for_loss = []
            label_for_loss = []

            
            pred_list.append(pred[:num_iter])
            label_list.append(label[:num_iter])
        

        
        
        preds_batch = torch.cat(pred_list)
        labels_batch = torch.cat(label_list)
        
        
        loss = nn.functional.cross_entropy(preds_batch, labels_batch, size_average=True)
        #print (loss)
        #input ('')
        return loss
def lev_error(str1, str2):
    
    str1= str1.replace(' ','')
    str2= str2.replace(' ','')

    rval= Lev.distance(str1,str2) 
    #s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
    #print (str2,rval)
    #input ('')
    return rval
class LAS(nn.Module):
    def __init__(self):
        super(LAS,self).__init__()
        self.listener = Listener_Block(40,256)
        self.attention = Attention_Block(key_query_dim=128, speller_query_dim=256, listener_feature=512, context_dim=128)
        #self.attention1= Attention()
        self.speller = Speller(33, 256, 3, self.attention, context_dim=128)
        print (self.speller)
        

    def forward(self,frames,sequence_length, labels, max_iter=250):
        
        listener_features, out_seq_sizes = self.listener(frames, sequence_length)
        #----batch size x updated_sequence  x feature_length=512

        
        outputs, attentions = self.speller(listener_features, out_seq_sizes, max_iter, ground_truth= labels,teacher_force_rate=0.8, dropout=[0.1, 0.1, 0.3])
        

        return outputs,attentions
def greedy_search(probs):
    # probs: FloatTensor (B, L, C)
    out = []

    for prob in probs:
        s = []
        
        for step in prob:
            #             idx = torch.multinomial(step, 1)[0]
            
            _,idx = step.max(0).values,step.max(0).indices#[1][0]
            c = letter_list[idx]

            s.append(c)
            if c == '<eos>':
                break
        out.append("".join(s))
    return out
def labels2str(labels, label_sizes):
    output = []
    for l, s in zip(labels, label_sizes):
        
        
        output.append("".join(letter_list[i] for i in l[:s]))
    return output
def train(model,optimizer,criterion,loader,epoch):

    model.train()
    total_loss=0.0
    batches=0.0

    for i, (frames, seq_sizes,labels,label_sizes) in enumerate(loader):

        optimizer.zero_grad()
        #frames = Length of sequence x Batch x Dimensions
        frames=Variable(frames.to(device))
        labels=Variable(labels.to(device))

        #print("Input shape from dataloader",frames.shape)
        
        output, attention= model(frames, seq_sizes,labels)



        #print ("Output in train=",output.shape,labels.shape,label_sizes)
        loss= criterion(output,label_sizes,labels)

        loss.backward()
        total_loss+= loss.item()
        optimizer.step()

        if i % 10 == 0:
            print ("Training loss ", loss.item())
        
        batches=batches+1

    return (total_loss/batches)

def compare_strings(original_string_list,decoded_string_list):

    batch_error=0
    for str_original,str_decoded in zip (original_string_list,decoded_string_list):

        error_distance= lev_error(str_original,str_decoded)

        batch_error = batch_error + error_distance

    return batch_error

def valid (model,optimizer,criterion,loader,epoch):

    model.eval()
    total_error=0.0
    total_loss = 0.0
    
    for i, (frames, seq_sizes,labels,label_sizes) in enumerate(loader):

        
        #frames = Length of sequence x Batch x Dimensions
        frames=Variable(frames.to(device))
        labels=Variable(labels.to(device))

        #print("Input shape from dataloader",frames.shape)
        
        output, attention= model(frames, seq_sizes,labels)



        #print ("Output in train=",output.shape,labels.shape,label_sizes)
        loss= criterion(output,label_sizes,labels)
        
        out_max=F.softmax(output).data.cpu()
        decoded= greedy_search(out_max)
        labels_str = labels2str(labels.data.cpu().numpy(), label_sizes)

        batch_error=compare_strings(labels_str,decoded)
        total_error = total_error + batch_error
        print ("Batch_erro",batch_error)
        input ('')
        '''
        for str1,str2 in zip (labels_str,decoded):
            
            error= cer (str1,str2)
            print ("Error =", error)
            input ('')
            total_error = total_error + error
        '''
        print (batch_error/len(label_sizes))
        total_loss+= loss.item()
        

        
    
    total_labels=loader.dataset.total_labels
    total_error = (total_error * 100.0 )/total_labels
    total_loss= (total_loss) / i
    return total_loss, total_error

def  predict (csv_fpath, weights_fpath,loader):

    model=LAS()
    model.load_state_dict(torch.load(weights_fpath))

    model.to(device)
    model.eval()
   
    with open(csv_fpath,"w") as csvfile:
        writer=csv.DictWriter(csvfile,fieldnames=['Id','Predicted'])
        writer.writeheader()
        count=0
        for batch , (frames,    seq_sizes, labels, label_sizes)  in enumerate(loader):
            frames=Variable(frames).to(device)
            labels=Variable(labels).to(device)

            outputs,attention= model(frames,    seq_sizes, None)
           
            decoded=greedy_search(F.softmax(outputs,dim=1).data.cpu())
            for s in decoded:
                print (s)


            for s in decoded:
                writer.writerow({'Id': count, 'Predicted':s})
                count=count+1
    print ("Over")

train_data_filepath='/etc/sanch_folder/cmupart4/train_new.npy'
train_label_filepath='/etc/sanch_folder/cmupart4/train_transcripts.npy'
valid_data_filepath='/etc/sanch_folder/cmupart4/dev_new.npy'
valid_label_filepath='/etc/sanch_folder/cmupart4/dev_transcripts.npy'

test_filepath='/etc/sanch_folder/cmupart4/test_new.npy'

model_dir='./model/'

#------------------------Setting up parameters or the environment -----------------------------#
# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda:3" if use_cuda else "cpu")   # use CPU or GPU

cores=4
#-------Training Parameters -------------#
batch_size=10
learning_rate=0.001
weight_decay=0.0001
number_of_epochs=20

# Data loading parameters
train_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': cores, 'pin_memory': True} if use_cuda else {}

valid_params = {'batch_size': batch_size, 'shuffle': True,'num_workers': cores, 'pin_memory': True} if use_cuda else {}


test_params = {'batch_size': 1, 'shuffle': False,'num_workers': cores, 'pin_memory': True} if use_cuda else {}


train_data=np.load(train_data_filepath,allow_pickle=True,encoding='latin1')

train_label=convert_bstring_to_normal(np.load(train_label_filepath,allow_pickle=True))
#train_label=np.load(train_label_filepath,allow_pickle=True,encoding='bytes')


valid_data=np.load(valid_data_filepath,allow_pickle=True,encoding='latin1')
valid_label=convert_bstring_to_normal(np.load(valid_label_filepath,allow_pickle=True))

test_data=np.load(test_filepath,allow_pickle=True,encoding='latin1')

train_label=transform_letter_to_index(train_label)
valid_label=transform_letter_to_index(valid_label)



'''Setting up dataset'''
train_set= Language_Dataset (train_data,train_label)
valid_set= Language_Dataset(valid_data,valid_label)
test_set = Language_Dataset(test_data,None)



'''Setting up dataloader'''
train_loader= DataLoader( train_set, collate_fn=collate_function, **train_params)
valid_loader= DataLoader( valid_set, collate_fn=collate_function, **valid_params)
test_loader = DataLoader( test_set,collate_fn=collate_function, **test_params)

#model=Listener_Block(input_dimension=40,hidden_dimension=256,dropout=0.2).to(device)

model=LAS().to(device)

#model=0
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#optimizer=0
criterion=Custom_Loss().to(device)


for e in range (number_of_epochs):
    #train(model=model,optimizer=optimizer,criterion=criterion,loader=train_loader,epoch=e)
    loss,error=valid(model=model,optimizer=optimizer,criterion=criterion,loader=valid_loader,epoch=e)

    print (loss,error)
    input ('')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    weight_fname = "{}/{:03d}_{}.w".format(model_dir, e, "{:.4f}".format(error))
    print("saving as", weight_fname)

    torch.save(model.state_dict(), weight_fname)
    input ('Here')



'''
weight_path='./model/006_58.8122.w'
predict ('submission.csv',weight_path,test_loader)
'''