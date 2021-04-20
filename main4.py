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
import Levenshtein as Lev

from speller2 import *


letter_list = ['<eos>', ' ', "'", '+', '-', '.','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',\
             'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','_','<sos>']

letter_to_index= {'<eos>': 0, ' ': 1, "'": 2, '+': 3, '-': 4, '.': 5, 'A': 6, 'B': 7, 'C': 8, 'D': 9, 'E': 10, 'F': 11, \
                'G': 12, 'H': 13, 'I': 14, 'J': 15, 'K': 16, 'L': 17, 'M': 18, 'N': 19, 'O': 20, 'P': 21, 'Q': 22, 'R': 23,\
                 'S': 24, 'T': 25, 'U': 26, 'V': 27, 'W': 28, 'X': 29, 'Y': 30, 'Z': 31, '_': 32,'<sos>':33}


#assert len(letter_list) == len(letter_to_index)

def transform_letter_to_index(transcript):

    output_transcript=[]
   
    for idx, list_of_words in enumerate(transcript): #list_of_wowrds = ['THE' 'INK' 'IS' 'ON' 'THE' 'WALL']
        
        char_sequence=['<sos>']
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
            return torch.tensor(frames[:-1].astype(float32)), [-1]
            
        else:
            
            return torch.from_numpy(frames).float(), (torch.from_numpy(self.y[idx][:-1]), torch.from_numpy(self.y[idx][1:]))
            
       
    def __len__(self):
        return self.x.shape[0]  

def collate_function(batch):
    batch_size = len(batch)
    
    batch = sorted(batch, key=lambda b: b[0].size(0), reverse=True)  # sort the batch by seq_len desc
    
    

    max_seq_len = batch[0][0].shape[0]    
    
    dimensions = batch[0][0].shape[1]
    

    pad_matrix = torch.zeros(max_seq_len, batch_size, dimensions) #----Sequence Length, Batch size , Vectors

    
    #labels=torch.zeros(batch_size,max_seq_len)

    max_length=max(l1.size(0) for (i,(l1,l2)) in batch)
    all_labels_1= torch.zeros(batch_size,max_length)
    all_labels_2= torch.zeros(batch_size,max_length)
    
    length_of_sequence = []
    label_sizes = torch.zeros(batch_size).int()
    

    for index, (frames, (labels1,labels2)) in enumerate(batch):
        number_of_frames = frames.shape[0]
        length_of_sequence.append(number_of_frames)

        labele_size = labels1.shape[0]
        label_sizes[index] = labele_size

        pad_matrix[:number_of_frames, index, :] = frames
        all_labels_1[index, :labele_size] = labels1  #Padding with labels only till the labele_size , the rest will be 0
        all_labels_2[index, : labele_size] = labels2


    return pad_matrix, length_of_sequence, all_labels_1.long(), all_labels_2.long(), label_sizes
def collate_test(batch):
    batch_size = len(batch)
    
    batch = sorted(batch, key=lambda b: b[0].size(0), reverse=True)  # sort the batch by seq_len desc
    
    

    max_seq_len = batch[0][0].shape[0]    
    pad_matrix = torch.zeros(max_seq_len, batch_size, dimensions) #----Sequence Length, Batch size , Vectors
        
    dimensions = batch[0][0].shape[1]
    for index, (frames, _) in enumerate(batch):
        number_of_frames= frames.shape[0]
        length_of_sequence.append(number_of_frames)
        pad_matrix[:number_of_frames,index,:] = frames


    return pad_matrix,length_of_sequence
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


def unpack_tensor( preds, label_sizes, labels):
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
    
    
    #loss = nn.functional.cross_entropy(preds_batch, labels_batch, size_average=False)
    #loss = criterion1(preds_batch,labels_batch)
    
    return preds_batch,labels_batch

def lev_error(str1, str2):
    
    a1= str1.replace(' ','')
    a2= str2.replace(' ','')
    

    rval= Lev.distance(str1,str2)
    #s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
    #print (str2,rval)
    #input ('')
    return rval


def greedy_search(output):
    # output = batches X seq_len X 33
    #print (output.shape)
    #input ('')
    output_string = []

    for items in output:
        item_string = []
        #items is seq len x 33
        
        for timestep in items:
            # timestep is 1 x 33
            #print (step)
            
            value,idx = timestep.max(0).values,timestep.max(0).indices
            character = letter_list[idx]
            if character == '<eos>':
                break
            item_string.append(character)

            
        

        output_string.append("".join(item_string))
  
    return output_string
def indices_to_string(labels, label_sizes):
    output_string = []
    
    for l, size in zip(labels, label_sizes):
        


        output_string.append("".join(letter_list[idx] for idx in l[:size]))
        
    return output_string

def train(model,optimizer,criterion,loader,epoch,teacher_force=0):

    model.train()
    total_loss=0.0
    batches=0.0
    total_error=0.0
    

    for i, (frames,seq_sizes,labels1,labels2,label_sizes) in enumerate(loader):

        optimizer.zero_grad()

        
        #frames = Length of sequence x Batch x Dimensions
        frames=frames.to(device)
        labels1=labels1.to(device)
        labels2=labels2.to(device)
       


        #print("Input shape from dataloader",frames.shape)
       
        output,_= model(frames, seq_sizes,labels1,train='train',teacher_force=teacher_force)
       
        mask= torch.zeros (labels2.size()).to(device)

        for idx in range (len (label_sizes)):
            mask[idx,:label_sizes[idx]]=1
        
        
        predictions= output.contiguous().view(-1,output.size(-1))
        labs= labels2.contiguous().view(-1)
        mask= mask.contiguous().view(-1)
      
        #print ("Output in train=",output.shape,labels.shape,label_sizes)
        #preds_batch,labels_batch= unpack_tensor( output, label_sizes , labels2)
        
        loss= criterion(predictions,labs)
        
        masked_loss= torch.sum (mask * loss) 

       
        masked_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)

        current_loss = float(masked_loss.item())/int(torch.sum(mask).item())
        
        total_loss+= current_loss

        '''
        output_max=F.softmax(output,dim=2).data.cpu()
        decoded= greedy_search(output_max)
        labels_str = indices_to_string(labels.data.cpu().numpy(), label_sizes)
        
        batch_error=compare_strings(labels_str,decoded)
        total_error = total_error + batch_error
        '''
        '''
        for s,original in zip(decoded,labels_str) :
            print ("Decoded=",s)
            print ("Original",original)
        '''
        optimizer.step()

        if i % 10 == 0:
            print ("Training loss ", current_loss)
            
        
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
    
    with torch.no_grad():
        for i, (frames,labels1,labels2,seq_sizes,label_sizes) in enumerate(loader):

            
            #frames = Length of sequence x Batch x Dimensions
            #frames=Variable(frames.to(device))
            #labels1=Variable(labels1.to(device))
            #labels2=Variable(labels2.to(device))

            frames=frames.to(device)
            labels1=labels1.to(device)
            labels2=labels2.to(device)
            
            #print("Input shape from dataloader",frames.shape)
            
            output,_= model(frames, seq_sizes,labels1,train='valid',teacher_force=0.0)


            mask= torch.zeros (labels2.size()).to(device)

            for idx in range (len (label_sizes)):
                mask[idx,:label_sizes[idx]]=1
        
        
            predictions= output.contiguous().view(-1,output.size(-1))
            labs= labels2.contiguous().view(-1)
            mask= mask.contiguous().view(-1)
      
            #print ("Output in train=",output.shape,labels.shape,label_sizes)
            #preds_batch,labels_batch= unpack_tensor( output, label_sizes , labels2)
        
            loss= criterion(predictions,labs)
        
            masked_loss= torch.sum (mask * loss) 
           
            current_loss = float(masked_loss.item())/int(torch.sum(mask).item())
            total_loss+= current_loss
           
            output_max=F.softmax(output,dim=2).data.cpu()
             
            decoded= greedy_search(output_max)
            labels_str = indices_to_string(labels2.data.cpu().numpy(), label_sizes)
            
            batch_error=compare_strings(labels_str,decoded)
            total_error = total_error + batch_error
            
            #print (batch_error/len(label_sizes))


def  predict (csv_fpath, weights_fpath,loader):

    model=LAS()
    model.load_state_dict(torch.load(weights_fpath))

    model.to(device)
    model.eval()
   
    with open(csv_fpath,"w") as csvfile:
        writer=csv.DictWriter(csvfile,fieldnames=['Id','Predicted'])
        writer.writeheader()
        count=0
        for batch , (frames,    seq_sizes)  in enumerate(loader):
            frames=Variable(frames).to(device)
            labels=Variable(labels).to(device)

            outputs,attention= model(frames,    seq_sizes, None,train='test')
           
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

model_dir='./model4/'

#------------------------Setting up parameters or the environment -----------------------------#
# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda:0" if use_cuda else "cpu")   # use CPU or GPU

cores=4
#-------Training Parameters -------------#
batch_size=64
learning_rate=0.0001
weight_decay=0
number_of_epochs=30

# Data loading parameters
train_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': cores, 'pin_memory': True} if use_cuda else {}

valid_params = {'batch_size': batch_size, 'shuffle': True,'num_workers': cores, 'pin_memory': True} if use_cuda else {}


test_params = {'batch_size': 1, 'shuffle': False,'num_workers': cores, 'pin_memory': True} if use_cuda else {}


train_data=np.load(train_data_filepath,allow_pickle=True,encoding='latin1')

train_label=convert_bstring_to_normal(np.load(train_label_filepath,allow_pickle=True,encoding='bytes'))
#train_label=np.load(train_label_filepath,allow_pickle=True,encoding='bytes')


valid_data=np.load(valid_data_filepath,allow_pickle=True,encoding='latin1')

valid_label=convert_bstring_to_normal(np.load(valid_label_filepath,allow_pickle=True,encoding='bytes'))
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
test_loader = DataLoader( test_set,collate_fn=collate_test, **test_params)


#model=Listener_Block(input_dimension=40,hidden_dimension=256,dropout=0.2).to(device)

model=Seq2Seq(input_dim=40,vocab_size=len(letter_list),encoder_hidden_dim=256,decoder_hidden_dim=256).to(device)

#model=0
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#optimizer=0
criterion=nn.CrossEntropyLoss(reduction='mean')
#criterion=Custom_Loss()

teacher_force=1.0
for e in range (number_of_epochs):

    print ("teacher_force",teacher_force)
    train(model=model,optimizer=optimizer,criterion=criterion,loader=train_loader,epoch=e,teacher_force=teacher_force)
    loss,error=valid(model=model,optimizer=optimizer,criterion=criterion,loader=valid_loader,epoch=e)

    print ("Loss and Error",loss,error)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    weight_fname = "{}/{:03d}_{}.w".format(model_dir, e, "{:.4f}".format(error))
    print("saving as", weight_fname)

    torch.save(model.state_dict(), weight_fname)
    
    if e %5 ==0 and teacher_force >=0.7:
        teacher_force= teacher_force -0.06

    print ('First epoch over')
    '''
    print ('20Th Iteration over')
    '''
'''
weight_path='./model/017_4.005.w'
predict ('submission.csv',weight_path,test_loader)
'''