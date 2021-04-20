import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.nn.utils.rnn import *
import pickle as pk
from torchnlp.nn import LockedDropout as lock_dropout
from torch.utils.data import DataLoader, Dataset
import time, pdb
import Levenshtein as Lev
import os
import csv
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print(device)

"""# **Pyramidal BiLSTM**


*   The length of utterance (speech input) can be hundereds to thousands of frames long.
*   Paper reports that that a direct LSTM implementation as Encoder resulted in slow convergence and inferior results even after extensive training.
*   The major reason is inability of `AttendAndSpell` operation to extract relevant information from a large number of input steps.
"""


letter_list = ['<eos>', ' ', "'", '+', '-', '.','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',\
             'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','_','<sos>']

letter_to_index= {'<eos>': 0, ' ': 1, "'": 2, '+': 3, '-': 4, '.': 5, 'A': 6, 'B': 7, 'C': 8, 'D': 9, 'E': 10, 'F': 11, \
                'G': 12, 'H': 13, 'I': 14, 'J': 15, 'K': 16, 'L': 17, 'M': 18, 'N': 19, 'O': 20, 'P': 21, 'Q': 22, 'R': 23,\
                 'S': 24, 'T': 25, 'U': 26, 'V': 27, 'W': 28, 'X': 29, 'Y': 30, 'Z': 31, '_': 32,'<sos>':33}
class pBLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, final_layer=False):
        super(pBLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        # nn.init.xavier_normal_(self.lstm1.all_weights)
        #TODO used dropout 0.2
        self.drop = nn.Dropout(0.3)
        self.final = final_layer

        for layer_p in self.lstm1._all_weights:
            for param in layer_p:
                if 'weight' in param:
                    nn.init.xavier_normal_(self.lstm1.__getattr__(param))

        # self.lstm2 = nn.LSTM(input_size=2*hidden_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        # self.lstm3 = nn.LSTM(input_size=2*hidden_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=True)





    def forward(self, x):
        '''
        :param x :(N,T) input to the pBLSTM
        :return output: (N,T,H) encoded sequence from pyramidal Bi-LSTM
        '''

        outputs, _ = self.lstm1(x)
        unpacked_seq, unpacked_lens = pad_packed_sequence(outputs, batch_first=True) # N, L, H
        # what if my unpacked_seq if of odd seq_len?
        if unpacked_seq.size(1) % 2 == 1:
            unpacked_seq = unpacked_seq[:, :-1, :]
        unpacked_seq = unpacked_seq.unsqueeze(2).reshape(unpacked_seq.size(0), unpacked_seq.size(1)//2, 2, unpacked_seq.size(2))
        unpacked_seq = unpacked_seq.mean(dim=2)

        if not self.final:
            unpacked_seq = self.drop(unpacked_seq)

        # unpacked_seq.transpose(1, 0).unsqueeze(2).reshape(unpacked_seq.size(0), unpacked_seq.size(1)//2, 2, unpacked_seq.size(2))
        # unpacked_seq.mean(dim=2).transpose(1, 0)
        packed_seq = pack_padded_sequence(unpacked_seq, unpacked_lens//2, batch_first=True, enforce_sorted=False)

        # outputs, _ = self.lstm2(packed_seq)
        # unpacked_seq, unpacked_lens = pad_packed_sequence(outputs, batch_first=True)
        # if unpacked_seq.size(1) % 2 == 1:
        #     unpacked_seq = unpacked_seq[:, :-1, :]
        # unpacked_seq = unpacked_seq.unsqueeze(2).reshape(unpacked_seq.size(0), unpacked_seq.size(1)//2, 2, unpacked_seq.size(2))
        # unpacked_seq = unpacked_seq.mean(dim=2)
        # packed_seq = pack_padded_sequence(unpacked_seq, unpacked_lens//2, batch_first=True, enforce_sorted=False)
        #
        #
        # outputs, _ = self.lstm3(packed_seq)
        # unpacked_seq, unpacked_lens = pad_packed_sequence(outputs, batch_first=True)
        # if unpacked_seq.size(1) % 2 == 1:
        #     unpacked_seq = unpacked_seq[:, :-1, :]
        # unpacked_seq = unpacked_seq.unsqueeze(2).reshape(unpacked_seq.size(0), unpacked_seq.size(1)//2, 2, unpacked_seq.size(2))
        # unpacked_seq = unpacked_seq.mean(dim=2)
        # # packed_seq = pack_padded_sequence(unpacked_seq, unpacked_lens)
        #
        # # unpacked_seq.transpose(1, 0).unsqueeze(1, 0).reshape(unpacked_seq.size(0), unpacked_seq.size(1)//2, 2, unpacked_seq.size(2))
        # # unpacked_seq.mean(dim=2).transpose(1, 0)
        # # packed_seq = pack_padded_sequence(unpacked_seq, unpacked_lens)

        return packed_seq, unpacked_lens//2


"""# **Encoder**

*    Encoder takes the utterances as inputs and returns the key and value.
*    Key and value are nothing but simple projections of the output from pBLSTM network.
"""


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, value_size=128, key_size=128):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        # nn.init.xavier_normal_(self.lstm.all_weights)
        self.dropout0 = lock_dropout(0.2)

        for layer_p in self.lstm._all_weights:
            for param in layer_p:
                if 'weight' in param:
                    nn.init.xavier_normal_(self.lstm.__getattr__(param))

        # Here you need to define the blocks of pBLSTMs
        self.pBLSTM1 = pBLSTM(input_dim=2*hidden_dim, hidden_dim=hidden_dim)
        self.dropout1 = lock_dropout(0.2)
        self.pBLSTM2 = pBLSTM(input_dim=2*hidden_dim, hidden_dim=hidden_dim)
        self.dropout2 = lock_dropout(0.2)
        self.pBLSTM3 = pBLSTM(input_dim=2*hidden_dim, hidden_dim=hidden_dim, final_layer=True)
        self.dropout3 = lock_dropout(0.2)

        self.key_network = nn.Linear(hidden_dim * 2, value_size)
        nn.init.xavier_normal_(self.key_network.weight)
        #TODO used dropout 0.2
        self.dropout4 = nn.Dropout(0.3)

        self.value_network = nn.Linear(hidden_dim * 2, key_size)
        nn.init.xavier_normal_(self.value_network.weight)

    def forward(self, x, lens):
        # X is N, T, F
        # x.permute(1, 0)
        
        rnn_inp = utils.rnn.pack_padded_sequence(x, lengths=lens, batch_first=True, enforce_sorted=False)

        out1, _ = self.lstm(rnn_inp)
        # out1, unpacked_lens = pad_packed_sequence(output, batch_first=True) # N, L, H
        out2, _ = self.pBLSTM1(out1)
        out3, _ = self.pBLSTM2(out2)
        out4, lens = self.pBLSTM3(out3)
        # outputs, _ = self.lstm(rnn_inp)
        # unpacked_seq = pad_packed_sequence(outputs)
        # out = self.pBLSTM(out)

        # Use the outputs and pass it through the pBLSTM blocks

        linear_input, _ = utils.rnn.pad_packed_sequence(out4, batch_first=True)
        linear_input = linear_input.transpose(1, 0)
        keys = self.key_network(linear_input)
        value = self.value_network(linear_input)
        
        return keys, value, lens


"""# **Attention**

*    Attention is calculated using key, value and query from Encoder and decoder.

Below are the set of operations you need to perform for computing attention.

```
energy = bmm(key, query)
attention = softmax(energy)
context = bmm(attention, value)
```
"""


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, lens):
        '''
        :param query :(N,context_size) Query is the output of LSTMCell from Decoder
        :param key: (N,key_size) Key Projection from Encoder per time step
        :param value: (N,value_size) Value Projection from Encoder per time step
        :return output: Attended Context
        :return attention_mask: Attention mask that can be plotted

        # Compute (N, T) attention logits. "bmm" stands for "batch matrix multiplication".
        # Input/output shape of bmm: (N, T, H), (N, H, 1) -> (N, T, 1)
        attention = torch.bmm(context, query.unsqueeze(2)).squeeze(2)
        # Create an (N, T) boolean mask for all padding positions
        # Make use of broadcasting: (1, T), (N, 1) -> (N, T)
        mask = torch.arange(context.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)
        # Set attention logits at padding positions to negative infinity.
        attention.masked_fill_(mask, -1e9)
        # Take softmax over the "source length" dimension.
        attention = nn.functional.softmax(attention, dim=1)
        # Compute attention-weighted sum of context vectors
        # Input/output shape of bmm: (N, 1, T), (N, T, H) -> (N, 1, H)
        out = torch.bmm(attention.unsqueeze(1), context).squeeze(1)


        '''

        energy = torch.bmm(key.transpose(1, 0), query.unsqueeze(2)).squeeze(2)
        mask = torch.arange(key.to('cpu').size(0)).unsqueeze(0) >= lens.to('cpu').unsqueeze(1)
        mask = mask.to(device)
        # set attention coefficient to large negative value for padded values along T
        energy.masked_fill_(mask, -1e9)
        attention_mask = nn.functional.softmax(energy, dim=1)
        attention_context = torch.bmm(attention_mask.unsqueeze(1), value.transpose(1, 0)).squeeze(1)
        # print('attention sum', torch.unique(torch.sum(attention_mask, dim=1)))
        return attention_context, attention_mask
        # attention_mask = energy.softmax()
        # attention_context = torch.bmm(attention_mask, value)
        # print('attention sum-', sum(attention_mask))
        # # attention_mask = torch.bmm(attention, )

        # return attention_context, attention_mask

"""# **Decoder**

*    As mentioned in Recitation-9 each forward call of decoder deals with just one time step. Thus we use LSTMCell instead of LSLTM here.
*    Output from the second LSTMCell can be used as query here for attention module.
*    In place of `value` that we get from the attention, this can be replace by context we get from the attention.
*    Methods like Gumble noise and teacher forcing can also be incorporated for improving the performance.
"""
def lev_error(str1, str2):
    
    a1= str1.replace(' ','')
    a2= str2.replace(' ','')
    

    rval= Lev.distance(str1,str2)
    #s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
    #print (str2,rval)
    #input ('')
    return rval


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=False):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 256)
        nn.init.xavier_normal_(self.embedding.weight)

        self.lstm1 = nn.LSTMCell(input_size=256 + value_size, hidden_size=hidden_dim)
        nn.init.xavier_normal_(self.lstm1.weight_hh)
        nn.init.xavier_normal_(self.lstm1.weight_ih)

        self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=key_size)
        nn.init.xavier_normal_(self.lstm2.weight_hh)
        nn.init.xavier_normal_(self.lstm2.weight_ih)

        self.isAttended = isAttended
        if (isAttended):
            self.attention = Attention()
        self.character_prob = nn.Linear(key_size + value_size, vocab_size)

    def forward(self, key, values, lens, text=None, train=True, teacher_prob = 0.9, val_max_len=250):
        '''
        :param key :(T,N,key_size) Output of the Encoder Key projection layer
        :param values: (T,N,value_size) Output of the Encoder Value projection layer
        :param text: (N,text_len) Batch input of text with text_length
        :param train: Train or eval mode
        :return predictions: Returns the character perdiction probability
        '''
        batch_size = key.shape[1]
        if (train):
            max_len = text.shape[1]
            embeddings = self.embedding(text)
        else:
            max_len = val_max_len # 250

        predictions = []
        hidden_states = [None, None]
        prediction = torch.zeros(batch_size, 1).to(device)
        draw_prob = torch.distributions.uniform.Uniform(0, 1)

        for i in range(max_len):
            '''
            Here you should implement Gumble noise and teacher forcing techniques
            '''
            if (train):

                # Teacher forcing
                if draw_prob.sample() >= 1-teacher_prob: # for sample > 0.1(for 0.9 teacher forcing) take GT
                    char_embed = embeddings[:, i, :]
                else:
                    # prediction = torch.distributions.gumbel.Gumbel(prediction, torch.Tensor(0.1))
                    char_embed = self.embedding(prediction.argmax(dim=-1))
            else:
                char_embed = self.embedding(prediction.argmax(dim=-1))

            # When attention is True you should replace the values[i,:,:] with the context you get from attention

            if i == 0:
                inp = torch.cat([char_embed, values[-1, :, :]], dim=1)
            else:
                inp = torch.cat([char_embed, attention_context], dim=1)

            # inp = torch.cat([char_embed, values[i, :, :]], dim=1)
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

            output = hidden_states[1][0]
            attention_context, attention_mask = self.attention(output, key, values, lens=lens)
            prediction = self.character_prob(torch.cat([output, attention_context], dim=1))
            # prediction = self.character_prob(torch.cat([output, values[-1, :, :]], dim=1))
            # prediction = self.character_prob(torch.cat([output, values[i, :, :]], dim=1))
            predictions.append(prediction.unsqueeze(1))

        return torch.cat(predictions, dim=1)


"""# **Sequence to Sequence Model**

*    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
"""


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, vocab_size, encoder_hidden_dim, decoder_hidden_dim, value_size=128, key_size=128, isAttended=False):
        super(Seq2Seq, self).__init__()

        self.encoder = Encoder(input_dim, encoder_hidden_dim)
        self.decoder = Decoder(vocab_size, decoder_hidden_dim, isAttended=True)

    def forward(self, speech_input, speech_len, text_input=None, train=True, teacher_prob=0.9, max_len=250):
        key, value, lens = self.encoder(speech_input, speech_len)
        if train:
            predictions = self.decoder(key, value, lens, text_input, train, teacher_prob)
        else:
            predictions = self.decoder(key, value, lens, text=None, train=False, val_max_len=max_len)
        return predictions


"""# **DataLoader**

Below is the dataloader for the homework.

*    You are expected to fill in the collate function if you use this code skeleton.
"""


class Speech2Text_Dataset(Dataset):
    def __init__(self, speech, text=None, train=True):
        self.speech = speech
        self.train = train
        if text is not None:
            self.text = text
            self.total_labels = sum(len(labels) for labels in text)
        else: 
            self.total_labels= -1
        
    def __len__(self):
        return self.speech.shape[0]

    def __getitem__(self, index):
        if self.train:
            # pdb.set_trace()
            # speech data, [<sos> Hi there], [Hi there <eos>] format
            speech_data = torch.tensor(self.speech[index].astype(np.float32))
            decoder_labels = torch.tensor(self.text[index][:-1])
            target_labels = torch.tensor(self.text[index][1:])
            return speech_data, (decoder_labels, target_labels)
            # return torch.tensor(self.speech[index].astype(np.float32)), torch.tensor(self.text[index])
        else:
            return torch.tensor(self.speech[index][:-1].astype(np.float32))


def collate_train(batch_data):
      '''
      Complete this function.
      I usually return padded speech and text data, and length of
      utterance and transcript from this function
      '''
      # pdb.set_trace()

      data, labels = zip(*batch_data)
      decoder_labels, target_labels = zip(*labels)
      X = [torch.FloatTensor(utterance) for utterance in data]
      Y_decoder = [torch.LongTensor(letter) for letter in decoder_labels]
      Y_target = [torch.LongTensor(letter) for letter in target_labels]
      X_lens = torch.LongTensor([len(seq) for seq in data])
      Y_lens = torch.LongTensor([len(seq) for seq in decoder_labels]) # both decoder_labels and target_labels have same lens and padding
      X = pad_sequence(X, batch_first=True)
      Y_decoder = pad_sequence(Y_decoder, batch_first=True)
      Y_target = pad_sequence(Y_target, batch_first=True)

      return X, Y_decoder, Y_target, X_lens, Y_lens


def collate_test(batch_data):
    '''
    Complete this function.
    I usually return padded speech and length of
    utterance from this function
    '''
    # data = zip(*batch_data)

    X = [torch.FloatTensor(utterance) for utterance in batch_data]
    X_lens = torch.LongTensor([len(seq) for seq in batch_data])
    X = pad_sequence(X, batch_first=True)
    # `batch_first=True` is required for use in `nn.CTCLoss`.

    return X, X_lens

def model_save(epoch ,model, optimizer, loss, name, PATH='./models/'):

    PATH = PATH + name + '.pth'

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, PATH)

    return

#assert len(letter_list) == len(letter_to_index)

def transform_letter_to_index(transcript):

    output_transcript=[]
   
    for idx, list_of_words in enumerate(transcript): #list_of_wowrds = ['THE' 'INK' 'IS' 'ON' 'THE' 'WALL']
        
        char_sequence=['<eos>']
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
    

    for i, (frames,labels1,labels2,seq_sizes,label_sizes) in enumerate(loader):

        optimizer.zero_grad()

        
        #frames = Length of sequence x Batch x Dimensions
        frames=frames.to(device)
        labels1=labels1.to(device)
        labels2=labels2.to(device)
       
        
        #print("Input shape from dataloader",frames.shape)
       
        output= model(frames, seq_sizes,labels1,train=True,teacher_prob=teacher_force)
        
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
            
            output= model(frames, seq_sizes,labels1,teacher_prob=0.0)


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
            '''            
            for s,original in zip(decoded,labels_str) :
                print ("Decoded=",s)
                print ("Original",original)
            '''
            '''
            for str1,str2 in zip (labels_str,decoded):
                
                error= cer (str1,str2)
                print ("Error =", error)
                input ('')
                total_error = total_error + error
            '''
            #print (batch_error/len(label_sizes))

           
        

            
  
    total_labels=loader.dataset.total_labels

    total_error = (total_error * 100.0 )/total_labels
    total_loss= (total_loss) / i
    return total_loss, total_error

def  predict (csv_fpath, weights_fpath,loader):

    model=model=Seq2Seq(input_dim=40,vocab_size=len(letter_list),encoder_hidden_dim=256,decoder_hidden_dim=512).to(device)
    model.load_state_dict(torch.load(weights_fpath))

    model.to(device)
    model.eval()
   
    with torch.no_grad():
        with open(csv_fpath,"w") as csvfile:
            writer=csv.DictWriter(csvfile,fieldnames=['Id','Predicted'])
            writer.writeheader()
            count=0
            for batch , (frames,    seq_sizes)  in enumerate(loader):
                
                
                    
                #frames=Variable(frames).to(device)
                
                outputs= model(frames,    seq_sizes, train=False)
                
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

model_dir='./model_x1/'

#------------------------Setting up parameters or the environment -----------------------------#
# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda:0" if use_cuda else "cpu")   # use CPU or GPU
#device= torch.device ("cpu")

print (device)
cores=2
#-------Training Parameters -------------#
batch_size=64
learning_rate=0.001
weight_decay=0
number_of_epochs=50

# Data loading parameters
train_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': cores, 'pin_memory': True} if use_cuda else {}

valid_params = {'batch_size': batch_size, 'shuffle': True,'num_workers': cores, 'pin_memory': True} if use_cuda else {}


test_params = {'batch_size': 1, 'shuffle': False,'num_workers': cores, 'pin_memory': True}if use_cuda else {}


train_data=np.load(train_data_filepath,allow_pickle=True,encoding='latin1')

train_label=convert_bstring_to_normal(np.load(train_label_filepath,allow_pickle=True,encoding='bytes'))
#train_label=np.load(train_label_filepath,allow_pickle=True,encoding='bytes')


valid_data=np.load(valid_data_filepath,allow_pickle=True,encoding='latin1')

valid_label=convert_bstring_to_normal(np.load(valid_label_filepath,allow_pickle=True,encoding='bytes'))
test_data=np.load(test_filepath,allow_pickle=True,encoding='latin1')

train_label=transform_letter_to_index(train_label)
valid_label=transform_letter_to_index(valid_label)



'''Setting up dataset'''
train_set= Speech2Text_Dataset (train_data,train_label)
valid_set= Speech2Text_Dataset(valid_data,valid_label)
test_set = Speech2Text_Dataset(test_data,None,False)


'''Setting up dataloader'''
train_loader= DataLoader( train_set, collate_fn=collate_train, **train_params)
valid_loader= DataLoader( valid_set, collate_fn=collate_train, **valid_params)
test_loader = DataLoader( test_set,collate_fn=collate_test, **test_params)


#weights_path='./model_c/004_63.2710.w'
model=Seq2Seq(input_dim=40,vocab_size=len(letter_list),encoder_hidden_dim=256,decoder_hidden_dim=512).to(device)

#model.load_state_dict(torch.load(weights_path))
#model=0
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#optimizer=0
criterion=nn.CrossEntropyLoss(reduction='none')
#criterion=Custom_Loss()

teacher_force=0.8

for e in range (number_of_epochs):
    if e > 24:
        learning_rate=0.0001

    print ("teacher_force",teacher_force)
    train(model=model,optimizer=optimizer,criterion=criterion,loader=train_loader,epoch=e,teacher_force=teacher_force)
    loss,error=valid(model=model,optimizer=optimizer,criterion=criterion,loader=valid_loader,epoch=e)

    print ("Loss and Error",loss,error)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    weight_fname = "{}/{:03d}_{}.w".format(model_dir, e, "{:.4f}".format(error))
    print("saving as", weight_fname)

    torch.save(model.state_dict(), weight_fname)
    
    
    print ('20Th iteration onwards')

'''
weight_path='./model_c2/017_8.9757.w'
predict ('submission.csv',weight_path,test_loader)
'''