speller2.py                                                                                         0000700 ?   U?8??   U?8?00000026513 13574471261 012752  0                                                                                                    ustar   sas479                          sas479                                                                                                                                                                                                                 import os
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

#letter_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',\
#             'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<eos>']
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
        self.lstm2 = nn.LSTMCell(hidden_size,key_size)

        #self.attention = Attention_Block(key_query_dim=128, speller_query_dim=hidden_size, listener_feature=512, context_dim=128)
        self.isAttended=isAttended
        if self.isAttended:
            self.attention= Attention()

        
        #self.lstm_inith= nn.ParameterList()
        #self.lstm_initc= nn.ParameterList()
        #self.lstm_inith.append(torch.nn.Parameter(torch.rand(1,hidden_size)))
        #self.lstm_inith.append(torch.nn.Parameter(torch.rand(1,key_size)))

        #self.lstm_initc.append(torch.nn.Parameter(torch.rand(1,hidden_size)))
        #self.lstm_initc.append(torch.nn.Parameter(torch.rand(1,key_size)))
        #self.attention= Attention()
        " Heree value size is same as the context size"
        self.character_prob = nn.Linear(key_size+value_size, vocab_size)
       
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
        hidden_states=[None,None]
        
        
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

            output=hidden_states[1][0]
            #print (output.shape)
            attention, context= self.attention(output, keys, values, seq_lengths)
            
            prediction = self.character_prob ( torch. cat([  output, context], dim=1))

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
      #predictions,attention = self.decoder(output, seq_lengths=seq_lengths , text=text_input,train=True,teacher_force_rate=teacher_force)
   
    elif train =='valid':
      predictions,attention = self.decoder(key,value,seq_lengths,text_input,train=False)
      #predictions,attention = self.decoder(output, seq_lengths=seq_lengths , text=text_input,train=False)
    elif train == 'test':
      predictions,attention = self.decoder(key,value,seq_lengths, text=None, train=False)
    else:
      print ('Invalid mode')
    
   
    return predictions,attention


                                                                                                                                                                                     result3.txt                                                                                         0000664 ?   U?8??   U?8?00000241513 13574075307 013023  0                                                                                                    ustar   sas479                          sas479                                                                                                                                                                                                                 teacher_force 1.0
Training loss  3.5252225093639837
Training loss  3.4313056041542587
Training loss  3.095659853348439
Training loss  3.03705838043942
Training loss  2.952258722716776
Training loss  2.9409067543451273
Training loss  2.929675902576888
Training loss  2.9291554352461016
Training loss  2.907859976327198
Training loss  2.927151803973123
Training loss  2.930323905109489
Training loss  2.908835685483871
Training loss  2.90267813402434
Training loss  2.8883542619319806
Training loss  2.8863640170367533
Training loss  2.869890764337331
Training loss  2.8415961116675876
Training loss  2.826474186089382
Training loss  2.735644380312983
Training loss  2.7078463835507756
Training loss  2.695234274711168
Training loss  2.6549980332326832
Training loss  2.6563164579395084
Training loss  2.6093116470530062
Training loss  2.644364639142082
Training loss  2.589367206280048
Training loss  2.594007843821592
Training loss  2.5587030258156185
Training loss  2.538431673498259
Training loss  2.508354921221234
Training loss  2.477491404170364
Training loss  2.4809045688587936
Training loss  2.4766019264363353
Training loss  2.4713571346507353
Training loss  2.4535939467371484
Training loss  2.4620701141539727
Training loss  2.42421500338824
Training loss  2.44045355145192
Training loss  2.432238535566246
Training loss  2.385348422427653
Training loss  2.4022284613715277
Training loss  2.416864285926131
Training loss  2.391333641409673
Training loss  2.392136895433874
Training loss  2.3878692077536825
Training loss  2.3495746574134118
Training loss  2.3984078982267354
Training loss  2.3375308388157894
Training loss  2.366436274168523
Training loss  2.34291179726112
Training loss  2.3641526464440927
Training loss  2.3403527780894673
Training loss  2.327074584777041
Training loss  2.3532840081849726
Training loss  2.3483686435722304
Training loss  2.3108939528810004
Training loss  2.33906232097846
Training loss  2.3389171565365277
Training loss  2.3156502125739387
Training loss  2.3159784637721237
Training loss  2.324620609789823
Training loss  2.2793709758377183
Training loss  2.270649985083532
Training loss  2.284211303852715
Training loss  2.30737917877907
Training loss  2.2655890701350963
Training loss  2.274499965317741
Training loss  2.1931115320994277
Training loss  2.2354230805906714
Training loss  2.2313478927773995
Training loss  2.222074304736635
Training loss  2.240286929733366
Training loss  2.2739610180412373
Training loss  2.20090808241634
Training loss  2.2707471989547536
Training loss  2.245305056688262
Training loss  2.2350387596899224
Training loss  2.1852899855172976
main2.py:272: UserWarning: torch.nn.utils.clip_grad_norm is now deprecated in favor of torch.nn.utils.clip_grad_norm_.
  torch.nn.utils.clip_grad_norm(model.parameters(), 2)
Loss and Error 4.394999780691039 156.96928849867027
saving as ./model6//000_156.9693.w
Epoch Over
teacher_force 0.94
Training loss  2.2484466272922212
Training loss  2.2120115444157245
Training loss  2.200276845391651
Training loss  2.21487735701196
Training loss  2.249376185054579
Training loss  2.2899439856226964
Training loss  2.2152693121072193
Training loss  2.194194649433817
Training loss  2.203587386372658
Training loss  2.204731883452786
Training loss  2.2168423852024213
Training loss  2.186493289884057
Training loss  2.1815304401131157
Training loss  2.1952363716269274
Training loss  2.162355369449488
Training loss  2.192889053520115
Training loss  2.160093443734305
Training loss  2.256194011742617
Training loss  2.16930069671975
Training loss  2.1691861051357892
Training loss  2.136843367993441
Training loss  2.1306301941582424
Training loss  2.140802716252619
Training loss  2.1124501329787235
Training loss  2.1062945364795547
Training loss  2.1445524409616845
Training loss  2.170928964617849
Training loss  2.1188975538270496
Training loss  2.1231384374010136
Training loss  2.164112009293127
Training loss  2.156232782125904
Training loss  2.15113748459976
Training loss  2.1801051999765106
Training loss  2.071073544357267
Training loss  2.160669367529247
Training loss  2.183386435055866
Training loss  4.11957036193056
Training loss  2.1222588922358763
Training loss  2.1286146294455284
Training loss  2.0824668455079927
Training loss  2.165502265861027
Training loss  2.1365363275645626
Training loss  2.124827762394515
Training loss  2.09571302928499
Training loss  2.13115397804426
Training loss  2.1031983464574435
Training loss  2.23499784932725
Training loss  2.1097580966350638
Training loss  2.0416836825284093
Training loss  2.103591209140933
Training loss  2.0910012989810247
Training loss  2.0542934974154137
Training loss  2.0681984560850015
Training loss  2.095783860907049
Training loss  2.0601491435592725
Training loss  2.11181493601539
Training loss  2.0977622700542824
Training loss  2.059475717487998
Training loss  2.018444562015215
Training loss  2.0717893926395745
Training loss  2.122887312313479
Training loss  2.054565351587492
Training loss  2.121949939751896
Training loss  2.0772950033004447
Training loss  2.0426295144217357
Training loss  2.0609641987825134
Training loss  2.0336375713195114
Training loss  2.048487988945578
Training loss  3.824853989684466
Training loss  2.163936175773627
Training loss  2.050777087329497
Training loss  2.077205822431328
Training loss  2.0442052387397993
Training loss  2.1245124278427245
Training loss  2.04179521188132
Training loss  2.00088012828142
Training loss  1.9977916165865384
Training loss  2.0346585190550983
Loss and Error 3.935550341383233 152.71867827607562
saving as ./model6//001_152.7187.w
Epoch Over
teacher_force 0.94
Training loss  2.078369757921144
Training loss  2.051926566926346
Training loss  2.0601122405297736
Training loss  2.019309907376625
Training loss  2.047399659579994
Training loss  2.0102238076979257
Training loss  2.0104079980677994
Training loss  2.0002065275377396
Training loss  1.984069588561776
Training loss  1.9968138570893759
Training loss  2.06218192673578
Training loss  2.0121633827473184
Training loss  2.002618180710055
Training loss  2.1285171031795
Training loss  2.0313357736013984
Training loss  2.059709643710191
Training loss  1.9992004683797393
Training loss  2.0365953157034853
Training loss  4.174116929996976
Training loss  2.0106529195988103
Training loss  2.0061209708651098
Training loss  2.0573249585700757
Training loss  2.005729756049791
Training loss  2.0266110233302124
Training loss  2.002194105673185
Training loss  1.9820579975451762
Training loss  1.984702171826571
Training loss  2.0721886320699596
Training loss  1.9866602264431716
Training loss  1.9623331397003034
Training loss  1.988399845001112
Training loss  1.969820069752695
Training loss  1.9781266385276846
Training loss  1.9734749748936673
Training loss  1.9421946861912953
Training loss  3.851325567634788
Training loss  1.9625990244636602
Training loss  1.9928164806059239
Training loss  1.965178809566145
Training loss  1.9307429283611197
Training loss  3.833541075600801
Training loss  1.9446641265368851
Training loss  3.5496242631621446
Training loss  1.9259093433956482
Training loss  1.9643908154632814
Training loss  3.5698425630179558
Training loss  1.9488359397651493
Training loss  1.9714558506592292
Training loss  1.9595692842211212
Training loss  1.892390091226708
Training loss  1.904318006617859
Training loss  1.9145703125
Training loss  2.0682003388116734
Training loss  1.951408713629714
Training loss  1.9616757079101261
Training loss  1.9585349834070795
Training loss  1.9376705858819345
Training loss  1.8988534788564204
Training loss  1.924507043878225
Training loss  1.950918224427283
Training loss  1.9633902383745734
Training loss  1.9529159489135948
Training loss  1.9763871851839143
Training loss  1.9886124663526246
Training loss  1.8618242704288899
Training loss  1.9292213348151097
Training loss  1.9178455334873827
Training loss  1.897688514888196
Training loss  1.9053906521644646
Training loss  1.9134344930702685
Training loss  1.9080757472826086
Training loss  1.92802484991657
Training loss  1.9725491936831552
Training loss  1.9694156434443508
Training loss  1.941450475103974
Training loss  1.9501131961892797
Training loss  1.9042989229141702
Training loss  1.9007475703595633
Loss and Error 3.788187078641475 149.4180115851215
saving as ./model6//002_149.4180.w
Epoch Over
teacher_force 0.94
Training loss  1.842059442824749
Training loss  1.9274120780840374
Training loss  1.9290337901297814
Training loss  3.541582235731172
Training loss  1.8943394540283285
Training loss  3.5413307250396198
Training loss  1.8752633489523944
Training loss  1.861323042046194
Training loss  3.499869558735838
Training loss  1.9298017358492183
Training loss  1.9224192795244235
Training loss  1.8338021161727616
Training loss  1.8722701551490017
Training loss  1.8917542399433485
Training loss  1.8680301029175197
Training loss  1.8434573139019537
Training loss  1.836738407332013
Training loss  1.847285179810012
Training loss  1.9159376209365324
Training loss  1.9502398530743819
Training loss  1.9049980293708442
Training loss  1.8859378826441366
Training loss  1.8226649404230268
Training loss  1.863323744999298
Training loss  1.8443324288922156
Training loss  1.870474382069365
Training loss  1.8327302222498578
Training loss  1.8005774864783655
Training loss  1.822284783719346
Training loss  1.894589397429192
Training loss  1.8343968670483461
Training loss  1.842409842888855
Training loss  1.8513169399527616
Training loss  1.8836167462707403
Training loss  1.948280736019737
Training loss  1.8611314620246326
Training loss  1.82075392045852
Training loss  1.8246410622173543
Training loss  1.8223049665178572
Training loss  1.8030963177447552
Training loss  3.337357121206195
Training loss  2.0965809165940077
Training loss  1.941283512636612
Training loss  1.9156809567789084
Training loss  1.8186410012393768
Training loss  1.881382893041237
Training loss  1.8742478466148975
Training loss  1.8827925975970807
Training loss  1.9109509581460895
Training loss  1.8665910465157038
Training loss  1.8414199709995664
Training loss  1.781589607341643
Training loss  1.894157918294271
Training loss  1.8714382288504001
Training loss  1.825317749634808
Training loss  1.8413092781924587
Training loss  3.084497388724598
Training loss  1.827374880946252
Training loss  1.904953731240849
Training loss  1.8376107616341992
Training loss  1.873912902745945
Training loss  1.86032097066986
Training loss  1.8132954234438536
Training loss  1.8285331291384355
Training loss  1.919187267292958
Training loss  1.85089599217496
Training loss  1.8370383630861753
Training loss  1.8140467845825297
Training loss  1.7820244733581165
Training loss  1.8634832477316317
Training loss  1.8035848716159375
Training loss  1.808047164059872
Training loss  1.7524409489150796
Training loss  1.771805605380173
Training loss  1.8295012885517934
Training loss  1.7384740005304784
Training loss  1.787623680273261
Training loss  1.7277519379844961
Loss and Error 3.0462822085618138 174.54552078399942
saving as ./model6//003_174.5455.w
Epoch Over
teacher_force 0.94
Training loss  1.772186241480793
Training loss  1.769433720206782
Training loss  1.75043479291366
Training loss  2.972304295193687
Training loss  1.8117650414209492
Training loss  1.8415719826587769
Training loss  1.8084626403533484
Training loss  2.952724717387768
Training loss  1.7719075324381413
Training loss  1.8419128705719618
Training loss  1.755249691299392
Training loss  1.7561259859366178
Training loss  1.8297248646160995
Training loss  1.7799422185910383
Training loss  1.733216758530695
Training loss  1.7807109804783008
Training loss  1.78617100130772
Training loss  1.7607773729605711
Training loss  1.7551750123031495
Training loss  1.7406084744608505
Training loss  1.77840872965412
Training loss  1.7470837953254899
Training loss  1.8268256904290932
Training loss  1.762053321188341
Training loss  1.730925779037454
Training loss  1.7485440164711512
Training loss  1.7381376458217945
Training loss  1.719205253085009
Training loss  1.740951789328348
Training loss  1.7300162897436862
Training loss  1.8015195469167782
Training loss  1.7648844258018452
Training loss  1.7750569525598219
Training loss  1.74863658389409
Training loss  1.7462112767524813
Training loss  1.7324833822513812
Training loss  1.743433494963056
Training loss  1.695420156105858
Training loss  1.6969272122524752
Training loss  1.7367973540273132
Training loss  1.7570877309657345
Training loss  1.7560289778600966
Training loss  1.6902665196730289
Training loss  1.7326514124368155
Training loss  1.7800558065935554
Training loss  1.7436821681725752
Training loss  2.9274015694932864
Training loss  1.814673345159409
Training loss  1.7483786364357203
Training loss  1.7395363962007262
Training loss  1.8044237729519774
Training loss  1.7754881925522252
Training loss  1.79558331011953
Training loss  1.7899875684716235
Training loss  1.7838081691576086
Training loss  1.7855271988263413
Training loss  1.7883972622826387
Training loss  1.6879158854166667
Training loss  1.7044376371021712
Training loss  2.9356034957785417
Training loss  1.8288495223309675
Training loss  1.6778507327621972
Training loss  1.7501785241374093
Training loss  1.7772159154587568
Training loss  1.7097488228025648
Training loss  1.7254639777918304
Training loss  1.766802011695133
Training loss  1.6926448134730199
Training loss  1.7316787591740448
Training loss  1.7273306630080978
Training loss  1.6771638725303952
Training loss  1.7243969240608354
Training loss  1.7493920180775098
Training loss  1.774018128576538
Training loss  1.7657876567476114
Training loss  1.685916785037879
Training loss  1.7883268720312937
Training loss  1.6283128294847045
Loss and Error 3.013001767522901 173.4252613938577
saving as ./model6//004_173.4253.w
Epoch Over
teacher_force 0.94
Training loss  2.932326363654716
Training loss  1.7095028524169593
Training loss  1.7080155269807156
Training loss  1.7794164941014057
Training loss  1.708288193350142
Training loss  1.7324767329677344
Training loss  1.7166905568222461
Training loss  1.6564785571808511
Training loss  1.711334500646895
Training loss  1.765307021722561
Training loss  1.6355127905795137
Training loss  1.7060709310545576
Training loss  1.7179967232880125
Training loss  1.6428373533612388
Training loss  1.6413682784537689
Training loss  1.6672776039923
Training loss  1.6502738332041882
Training loss  1.6736760731894762
Training loss  1.6556534162982024
Training loss  1.7662888053627648
Training loss  1.746851696670684
Training loss  1.7016239872685186
Training loss  1.7012174371645334
Training loss  1.6990072721797052
Training loss  1.6797400954992436
Training loss  1.7142317024930533
Training loss  2.950475313645334
Training loss  1.6706828845972013
Training loss  2.929536632306889
Training loss  1.6378382637469724
Training loss  1.6558065397036474
Training loss  1.7301429485452586
Training loss  1.6863461674337856
Training loss  1.6062914939180304
Training loss  1.7176218150898384
Training loss  1.6548399276551313
Training loss  1.7156218354711483
Training loss  2.96055626131501
Training loss  1.6835016130640383
Training loss  1.6516820111012895
Training loss  1.6874025033190359
Training loss  1.6150982446101594
Training loss  1.6387159592737979
Training loss  1.6565543574115216
Training loss  1.622416931027891
Training loss  1.7038357873477845
Training loss  1.6645520154522118
Training loss  1.6415726824504573
Training loss  1.6753235232296706
Training loss  1.6837255308977457
Training loss  1.6661974366359447
Training loss  1.761799197865521
Training loss  1.7197963382633588
Training loss  1.666883752059308
Training loss  1.6136065795754995
Training loss  1.5742230751739956
Training loss  1.6304381313899707
Training loss  1.7470615489015409
Training loss  2.9561469381836263
Training loss  2.9416402443393523
Training loss  1.6160813360699153
Training loss  1.631036728464496
Training loss  1.65631103515625
Training loss  1.6150138759887345
Training loss  1.5811598422503
Training loss  1.6910198425144465
Training loss  2.9413697730705852
Training loss  1.6829759025513045
Training loss  1.7019505193549531
Training loss  1.6645682572078617
Training loss  1.5926762685280305
Training loss  1.6155494600183824
Training loss  1.680675242003367
Training loss  1.6128104818264304
Training loss  1.6283128772350857
Training loss  1.5857904905913978
Training loss  1.6251635648338825
Training loss  1.6140269835676564
Loss and Error 3.0155195793663228 175.42715581624103
saving as ./model6//005_175.4272.w
Epoch Over
teacher_force 0.8799999999999999
Training loss  1.6419262192086976
Training loss  2.9240011369355767
Training loss  1.648705091651611
Training loss  1.5504517569783405
Training loss  1.6768150401701323
Training loss  1.5490336693079665
Training loss  2.9458648875795257
Training loss  1.6360741808620272
Training loss  1.703663843787628
Training loss  1.6313293616577114
Training loss  1.6446228567477876
Training loss  1.6908039978895142
Training loss  1.6115509999823248
Training loss  1.573165986032197
Training loss  1.581098573507085
Training loss  1.586714291078402
Training loss  1.5181284656550893
Training loss  1.6175118467275322
Training loss  1.6060531496062993
Training loss  1.696251227094241
Training loss  2.922788185841592
Training loss  2.935363306333917
Training loss  1.5947181854736114
Training loss  1.6457885962929475
Training loss  1.6214975555981594
Training loss  1.5806115949276116
Training loss  1.560443483143354
Training loss  2.930948138202969
Training loss  1.6009981180537607
Training loss  1.6491438206032192
Training loss  1.671589475896861
Training loss  1.5926124271953406
Training loss  1.6820800202717714
Training loss  1.5848021431291002
Training loss  1.6632326661683083
Training loss  1.6321639328168631
Training loss  2.9050775247101144
Training loss  1.6694498969636895
Training loss  1.5643691696128534
Training loss  1.6508316102489768
Training loss  1.567721454670526
Training loss  1.5803872105751025
Training loss  1.6616604506266384
Training loss  1.5386714791726983
Training loss  2.922134112953692
Training loss  1.5987204382753895
Training loss  1.6213739621314205
Training loss  1.6270029039590492
Training loss  1.5398571451969152
Training loss  1.4901944857525613
Training loss  1.6255145057337104
Training loss  1.6920395795036764
Training loss  1.4858882655383148
Training loss  1.579867034606754
Training loss  1.5770564646237484
Training loss  1.6152977103418054
Training loss  1.5810767262725678
Training loss  1.6221879098320544
Training loss  2.921576070452475
Training loss  1.5528526684727086
Training loss  1.5303648428064613
Training loss  1.5556397228087788
Training loss  1.5196059024775164
Training loss  2.917409985303723
Training loss  1.5478521596033323
Training loss  1.7276602083708978
Training loss  1.5381457240429453
Training loss  2.9312980025286834
Training loss  1.5719729468149852
Training loss  1.5597931585934985
Training loss  1.5915602295511384
Training loss  1.636461631123017
Training loss  1.5570443735001713
Training loss  1.5758817607328555
Training loss  1.6339398487170753
Training loss  1.6032717821074696
Training loss  1.5613462063068515
Training loss  1.5598502672751515
Loss and Error 3.0082418632451655 174.49633866443222
saving as ./model6//006_174.4963.w
Epoch Over
teacher_force 0.8799999999999999
Training loss  1.571141836056231
Training loss  1.574643417143486
Training loss  1.5492781368035884
Training loss  2.923425976795292
Training loss  1.5462391838166512
Training loss  1.6089055045324427
Training loss  2.933750854272768
Training loss  1.5747659254807693
Training loss  1.6038636995715652
Training loss  1.5517578125
Training loss  1.5481022556836121
Training loss  1.531708706019893
Training loss  1.5439412972650515
Training loss  1.5152565803113691
Training loss  1.609240868289232
Training loss  1.5958227882922535
Training loss  1.6063625951664307
Training loss  1.5167292854144574
Training loss  1.5861256125064902
Training loss  1.5643549388095417
Training loss  1.513444553122051
Training loss  1.6572801036306777
Training loss  1.5645882693594984
Training loss  1.6259498018366152
Training loss  1.5379868442481606
Training loss  1.5307072425956156
Training loss  1.6434887269527787
Training loss  1.503575038243007
Training loss  1.6201744151701323
Training loss  1.5895950418678335
Training loss  1.456981814775755
Training loss  1.5582614551110798
Training loss  1.4879816161962567
Training loss  1.4850026943356691
Training loss  2.9155854761809765
Training loss  1.549934786297415
Training loss  1.5121185606346095
Training loss  1.5268533074778685
Training loss  1.5378826144616917
Training loss  1.593988265782059
Training loss  1.5705475370762711
Training loss  1.569845401440315
Training loss  2.929212992947611
Training loss  1.561122594944668
Training loss  1.6145300812648853
Training loss  1.5852690944328407
Training loss  1.5656311543177766
Training loss  1.5153922182750466
Training loss  1.5586164746631965
Training loss  1.5534474043266702
Training loss  1.493265035598888
Training loss  1.5303922800852183
Training loss  1.5571397363157478
Training loss  1.5610759181180187
Training loss  1.4724362139864076
Training loss  1.5728777301377894
Training loss  1.5094655092241118
Training loss  1.594753283674189
Training loss  1.499693845742786
Training loss  1.5792324420103092
Training loss  1.5528226022000127
Training loss  1.5504699840670941
Training loss  2.9221041757880912
Training loss  1.5306761346670648
Training loss  1.6421819973056686
Training loss  2.919281316207627
Training loss  1.577986940713016
Training loss  1.6528661881662225
Training loss  1.5951858551440776
Training loss  1.6300477761033558
Training loss  1.5926607343369366
Training loss  1.5248786153459293
Training loss  2.9319628473321724
Training loss  1.502191148515799
Training loss  1.5276257897111913
Training loss  1.5218921392839924
Training loss  1.529310146682721
Training loss  1.4702226081983198
Loss and Error 3.0056356499686347 175.6985682538526
saving as ./model6//007_175.6986.w
Epoch Over
teacher_force 0.8799999999999999
Training loss  1.4271410662443917
Training loss  1.5284473869371118
Training loss  1.5313498217743773
Training loss  1.506391470837011
Training loss  1.5241246322390833
Training loss  1.5248081373131794
Training loss  1.526402711050103
Training loss  1.619741734899605
Training loss  2.944966980942425
Training loss  1.5336071909801543
Training loss  1.4791588024877769
Training loss  1.5186127409505592
Training loss  1.4721864337503554
Training loss  1.4674025301923632
Training loss  1.5545238545095164
Training loss  1.5728486654048208
Training loss  1.5435870519632564
Training loss  1.5287244923107255
Training loss  1.5301403686387653
Training loss  1.5728217274111977
Training loss  1.5083677062888452
Training loss  1.4420169664355695
Training loss  1.5250134447058326
Training loss  1.4923906283141262
Training loss  1.5225659283199575
Training loss  1.373485362426607
Training loss  1.5166336146868413
Training loss  1.4913947678518409
Training loss  1.4851482884323504
Training loss  1.5357229184144294
Training loss  1.5240072893744105
Training loss  1.4985473527670112
Training loss  1.4499160573093877
Training loss  1.527690881772849
Training loss  1.5739468936588865
Training loss  1.4568458764472407
Training loss  1.473037598123207
Training loss  1.551333880898179
Training loss  1.5517380758605817
Training loss  1.4951049767908255
Training loss  2.9330629355400695
Training loss  1.4376472107438016
Training loss  1.519334382464172
Training loss  1.4626808019596438
Training loss  1.4711746180252039
Training loss  1.52297797386202
Training loss  1.4785040073915325
Training loss  1.5699532810753702
Training loss  1.4917286981880171
Training loss  1.468387173026538
Training loss  1.5336698008849559
Training loss  1.5052385906454138
Training loss  2.9282403186989496
Training loss  1.5078886397271152
Training loss  1.4959151594606164
Training loss  1.4791243873940356
Training loss  1.4537172548854447
Training loss  1.5267732335442534
Training loss  1.4520213582765422
Training loss  1.4088725932679715
Training loss  1.4490595121182923
Training loss  1.4425206730021767
Training loss  1.4491417534983448
Training loss  1.4403635255403127
Training loss  1.4840810110992402
Training loss  1.4944366409732541
Training loss  1.532965770156722
Training loss  1.4817205632774575
Training loss  1.5590747985470828
Training loss  1.3827572142564721
Training loss  1.5223612828991653
Training loss  2.913216999258739
Training loss  1.4890054610363315
Training loss  1.519455085789323
Training loss  1.4848019860291708
Training loss  1.5244475399770068
Training loss  1.5003196922508446
Training loss  1.4132235004885807
Loss and Error 2.998924922079169 175.24955371780393
saving as ./model6//008_175.2496.w
Epoch Over
teacher_force 0.8799999999999999
Training loss  1.480707529207954
Training loss  1.4808609154206376
Training loss  2.8856833279325986
Training loss  1.4474196839395268
Training loss  1.5199031923396986
Training loss  1.4031140675164016
Training loss  1.4907631882506183
Training loss  2.9506380562882915
Training loss  2.9256184895833335
Training loss  1.4241150818787538
Training loss  1.4016030889209592
Training loss  1.4890220682457012
Training loss  1.4194317997975117
Training loss  1.58521077844711
Training loss  1.43189662265945
Training loss  1.4262992658423492
Training loss  1.5079035768792293
Training loss  1.3973444809506819
Training loss  1.3922260842559704
Training loss  1.4078715561320048
Training loss  1.4024904623558088
Training loss  1.5560577007363254
Training loss  1.4169339506333056
Training loss  2.9261609779746656
Training loss  1.518435759323969
Training loss  1.4452762586137204
Training loss  1.4959573004777569
Training loss  1.4289740737603636
Training loss  1.4002032235640771
Training loss  1.4102083127957412
Training loss  1.4447382731337193
Training loss  1.4865213879600208
Training loss  1.4732692513195065
Training loss  2.9489317143883413
Training loss  2.652923278808594
Training loss  2.4189650472647615
Training loss  2.2618602942634
Training loss  2.167502095679975
Training loss  2.135289420186178
Training loss  2.0790445089402674
Training loss  2.0105071159918277
Training loss  3.814365581530139
Training loss  2.0825312988135973
Training loss  2.0461612381030223
Training loss  2.0262918982132807
Training loss  1.9753723663243625
Training loss  2.9628011042505418
Training loss  1.922062675734073
Training loss  1.9789259453781514
Training loss  1.9196762062220567
Training loss  1.9038727442615524
Training loss  1.8283031314357163
Training loss  1.8993365372835675
Training loss  2.946306905338151
Training loss  2.9543209471288514
Training loss  2.9776703725961537
Training loss  2.925242394051322
Training loss  1.8279223074108568
Training loss  2.953623282191265
Training loss  1.8213682174682617
Training loss  1.735159669868677
Training loss  2.9347838240331914
Training loss  1.825064187703032
Training loss  1.7833572960923525
Training loss  1.7533492281009546
Training loss  1.6720515167364016
Training loss  1.7026494779959258
Training loss  1.6652497340125039
Training loss  1.6947392943828017
Training loss  2.9163329684983896
Training loss  1.745098898520318
Training loss  1.6824619535765897
Training loss  1.6568359943099185
Training loss  1.6482910657737075
Training loss  1.64680808104691
Training loss  1.6481911895686485
Training loss  1.667376476551481
Training loss  1.6757815482781002
Loss and Error 3.0091676863724963 173.94804910925717
saving as ./model6//009_173.9480.w
Epoch Over
teacher_force 0.8799999999999999
Training loss  1.6537620775099828
Training loss  1.677055854301948
Training loss  1.5980535679517134
Training loss  1.679417022737999
Training loss  2.9215165821271647
Training loss  1.5897245675184386
Training loss  1.6058007572999693
Training loss  1.5638047191295548
Training loss  1.6136985293444004
Training loss  1.7085623284936107
Training loss  1.6047846372889965
Training loss  1.5971472372555866
Training loss  1.55642887437761
Training loss  1.599924412356714
Training loss  1.6437673158409387
Training loss  1.5503059307085034
Training loss  1.5499117266085676
Training loss  1.5988047480296554
Training loss  1.5769092401047526
Training loss  1.6155531545645858
Training loss  1.5088792091178107
Training loss  1.5715178367047613
Training loss  1.5797865761308563
Training loss  1.5536052416162924
Training loss  1.5140735051190821
Training loss  1.5387076428004043
Training loss  1.5640162760416667
Training loss  1.5257893926684751
Training loss  1.5088980594758064
Training loss  2.897849693480861
Training loss  1.447798670272919
Training loss  1.562746253786786
Training loss  2.920453690406577
Training loss  2.940837633095099
Training loss  1.4427019571520618
Training loss  1.6129638042646586
Training loss  1.5142268249078372
Training loss  1.6013292176098854
Training loss  1.4869759084963603
Training loss  1.564162625087781
Training loss  1.5360614812182114
Training loss  1.4971023640782082
Training loss  1.5346320938252784
Training loss  1.5100558964022286
Training loss  1.4993383167369236
Training loss  1.531091146568766
Training loss  1.5577908503100477
Training loss  1.5552257701069732
Training loss  1.530963882173896
Training loss  1.4637038010817307
Training loss  1.4791482472779491
Training loss  1.5523869737736926
Training loss  1.4731353183962264
Training loss  1.5756988474135036
Training loss  1.590188520170839
Training loss  1.5586710069444445
Training loss  1.541147312973485
Training loss  1.5042753732214718
Training loss  1.5081637198464912
Training loss  1.5254694002080667
Training loss  1.5990652418219238
Training loss  1.4718145613732867
Training loss  1.4392448452188602
Training loss  1.445023678296233
Training loss  1.4805266142761864
Training loss  1.5557950577645931
Training loss  1.3886300945398142
Training loss  2.945106756233402
Training loss  1.4697785653868785
Training loss  1.4170732110010957
Training loss  1.3816387260419987
Training loss  1.4863558235334005
Training loss  1.4910857928240742
Training loss  1.4310293681400177
Training loss  1.4518060830444535
Training loss  1.4502686511857708
Training loss  1.4675706040458667
Training loss  1.4454775486132334
Loss and Error 3.0119968665506445 177.82979343509783
saving as ./model6//010_177.8298.w
Epoch Over
teacher_force 0.8199999999999998
Training loss  2.942437545213981
Training loss  1.4592428174534804
Training loss  1.4849191509394042
Training loss  2.933651089994676
Training loss  2.924229873686757
Training loss  1.4699422660787986
Training loss  1.3675343532650694
Training loss  1.434005908161068
Training loss  1.3772982972317798
Training loss  1.4261284297537509
Training loss  1.4760938240989567
Training loss  2.9140409169596726
Training loss  1.482068397102526
Training loss  1.419118872186019
Training loss  2.925624180506993
Training loss  2.9321822362588654
Training loss  1.4112479504243827
Training loss  1.4743876706091898
Training loss  1.437456680100617
Training loss  2.927348225245499
Training loss  1.5297957211312323
Training loss  1.522575514533565
Training loss  2.9065865489924065
Training loss  2.916988464898348
Training loss  1.5104589972839557
Training loss  1.464968255630455
Training loss  1.4487443411568213
Training loss  1.4574055738021638
Training loss  1.4186426860179198
Training loss  1.439173692759901
Training loss  1.4811104998235571
Training loss  1.4387899819906655
Training loss  2.9275530018600167
Training loss  1.4185499253087905
Training loss  1.3698899883169715
Training loss  1.389706107355378
Training loss  1.4172641063588263
Training loss  1.4020583684336374
Training loss  1.4140654029800832
Training loss  1.3429255500389408
Training loss  1.375900637084464
Training loss  2.8930305198386863
Training loss  1.4922489361756734
Training loss  2.9590953918370944
Training loss  1.4151635177149082
Training loss  2.937439467052935
Training loss  1.4600499288069544
Training loss  1.3556093671017475
Training loss  1.3823433527146276
Training loss  1.3895952987058589
Training loss  2.9195659549744537
Training loss  1.3649274600894674
Training loss  1.4687824271233205
Training loss  1.3844559868296828
Training loss  1.4024205913664456
Training loss  1.3874151918054771
Training loss  1.470302483974359
Training loss  1.4481858899167848
Training loss  1.4784701551087949
Training loss  1.4235269145601115
Training loss  1.3471027321749844
Training loss  1.432323027820122
Training loss  1.3712580726042292
Training loss  1.3639936447143555
Training loss  1.31290884864435
Training loss  1.5033282511483852
Training loss  1.3741576865211718
Training loss  1.4899348523382934
Training loss  1.429905332037386
Training loss  1.4859316972373189
Training loss  1.4498805835591329
Training loss  2.92262724384726
Training loss  1.324721792859842
Training loss  1.3769363679694948
Training loss  2.904359165424739
Training loss  1.4454373984913118
Training loss  1.4449598524305556
Training loss  1.353514419367284
Loss and Error 3.003905538464808 171.91063426718642
saving as ./model6//011_171.9106.w
Epoch Over
teacher_force 0.8199999999999998
Training loss  1.3535411688883299
Training loss  1.2768640245542575
Training loss  1.542245162869594
Training loss  1.3969681602621178
Training loss  1.3393902693288855
Training loss  2.919438263504931
Training loss  1.3426738726587948
Training loss  1.3743572549662668
Training loss  2.944909514085413
Training loss  1.3645275695931478
Training loss  1.3843097297512754
Training loss  1.360388923631548
Training loss  1.4144977259551679
Training loss  1.3738785695463773
Training loss  1.5108300308674922
Training loss  1.334529104949422
Training loss  2.893333902068594
Training loss  1.441039057668522
Training loss  1.2984096395013318
Training loss  1.3406477844649418
Training loss  1.264819682293385
Training loss  1.3031872526554298
Training loss  1.387095842887735
Training loss  1.374576579571593
Training loss  1.4024078753068425
Training loss  2.937242138441915
Training loss  2.9267578125
Training loss  1.3718976658278768
Training loss  1.482259314166922
Training loss  1.4356787551925982
Training loss  1.3529042488665317
Training loss  1.2872092298897748
Training loss  1.3317861500798134
Training loss  2.9235706160140715
Training loss  1.3519382877244475
Training loss  2.9177302308000583
Training loss  1.4784497943576123
Training loss  1.338755765545986
Training loss  1.4021935807743922
Training loss  1.4253795306307795
Training loss  2.9125781018243844
Training loss  1.4017417308820392
Training loss  1.4067052396616542
Training loss  1.3794461333613317
Training loss  1.267974353227459
Training loss  1.4038367888473497
Training loss  1.442655037469568
Training loss  2.9506952290468367
Training loss  1.3902962313818414
Training loss  1.40733000346513
Training loss  1.3058432095429278
Training loss  1.3317775843814168
Training loss  1.4519372088950318
Training loss  1.3874287855585548
Training loss  1.4034639415977554
Training loss  1.2710688470811249
Training loss  1.3786831098472268
Training loss  1.2664176387000379
Training loss  1.3624375190580837
Training loss  1.391631520299652
Training loss  1.4427806486623904
Training loss  1.4683730375596185
Training loss  1.4221687019865823
Training loss  1.4560957741938758
Training loss  1.377927943119232
Training loss  2.924305366299257
Training loss  1.428902767731467
Training loss  1.2750833445581897
Training loss  1.314316834904333
Training loss  1.4060067602629152
Training loss  1.3285808257603857
Training loss  1.3829604732226197
Training loss  1.4124039685037215
Training loss  2.8943420510708404
Training loss  2.9249579676961988
Training loss  1.3636500274618102
Training loss  1.2863613447739077
Training loss  1.2268911340032875
Loss and Error 2.9894545356514217 171.77401726838866
saving as ./model6//012_171.7740.w
Epoch Over
teacher_force 0.8199999999999998
Training loss  2.8963183902167673
Training loss  1.4208660179494206
Training loss  1.3149463939767523
Training loss  1.271093578221196
Training loss  2.889967371065205
Training loss  1.3896134876447814
Training loss  2.923838167471512
Training loss  1.3474757038198857
Training loss  1.381769427429402
Training loss  1.3591861404336372
Training loss  1.42579837231711
Training loss  2.9017927678827924
Training loss  1.381772730024756
Training loss  1.2788526492011278
Training loss  1.3614338876053826
Training loss  1.412992400085034
Training loss  2.9221500223747015
Training loss  1.3502854997672586
Training loss  1.330789780475684
Training loss  1.3312932401651907
Training loss  2.895734696719422
Training loss  1.4234178444225523
Training loss  1.304971632909521
Training loss  1.2682750056865901
Training loss  1.366314594972067
Training loss  1.370033665707237
Training loss  1.3756043853073463
Training loss  1.265114015758547
Training loss  1.281198391836387
Training loss  2.8836071925607287
Training loss  1.32962453130311
Training loss  1.3983853619901019
Training loss  1.3643983092613454
Training loss  1.3111277638450556
Training loss  1.3500528031318757
Training loss  1.3749770773773697
Training loss  1.3503698152805328
Training loss  1.356820522808963
Training loss  1.3137246238515288
Training loss  1.2243659101967994
Training loss  2.9186403250342816
Training loss  1.320601433834152
Training loss  1.3840391081109258
Training loss  1.3940236964258812
Training loss  1.2438864997320251
Training loss  1.3627067141418365
Training loss  1.3081673293886282
Training loss  1.3520301823380867
Training loss  2.901191841786128
Training loss  1.460036216030563
Training loss  1.3265517697350167
Training loss  1.3757916074810606
Training loss  1.3321662356015358
Training loss  1.3453549066371506
Training loss  1.3892031262365463
Training loss  1.2722701639524647
Training loss  1.405301429054548
Training loss  1.22006200542204
Training loss  1.2589612159019237
Training loss  1.3503770385981384
Training loss  1.2607510567020759
Training loss  1.3333526097156838
Training loss  1.3064266216153562
Training loss  1.309833347414667
Training loss  1.2510484753886493
Training loss  1.3932651358195212
Training loss  2.905803716876629
Training loss  1.320675691793893
Training loss  1.3421397596681832
Training loss  1.3566067319451596
Training loss  1.2456519225687104
Training loss  1.3475931310639773
Training loss  1.2756707035041248
Training loss  1.3365995569574096
Training loss  1.3152032910801494
Training loss  1.2139177865932642
Training loss  1.3017859150179856
Training loss  1.407782526299505
Loss and Error 3.0057276221536315 171.41790229152247
saving as ./model6//013_171.4179.w
Epoch Over
teacher_force 0.8199999999999998
Training loss  2.9431155477775994
Training loss  1.316731121831387
Training loss  1.3586534251110731
Training loss  1.351405983318433
Training loss  2.9103764523988005
Training loss  1.3174739889166178
Training loss  1.2802455200742713
Training loss  1.3659360638786764
Training loss  1.3182595876492922
Training loss  1.3127038707089032
Training loss  1.2740964668808237
Training loss  1.2932390303459589
Training loss  1.3097529567459067
Training loss  1.3472546817443776
Training loss  1.2535647425381033
Training loss  2.932646731022759
Training loss  1.34015924182983
Training loss  1.285332275688721
Training loss  1.202032225295882
Training loss  1.257502287849827
Training loss  2.8840216470378968
Training loss  1.2672386181440825
Training loss  1.3179529683702915
Training loss  1.2363547145730198
Training loss  1.2367712549416416
Training loss  2.9274552739036395
Training loss  1.352389447723523
Training loss  1.294381872755835
Training loss  1.2715788634298097
Training loss  1.319247692041864
Training loss  1.2395156198244452
Training loss  1.2687810189007753
Training loss  1.2985606557419542
Training loss  1.3957672299291617
Training loss  1.3357533688763885
Training loss  1.2989580454933682
Training loss  1.339685828528758
Training loss  1.2536665139739047
Training loss  1.3472095304573
Training loss  1.231001738293152
Training loss  2.943508031466958
Training loss  2.9079936264557156
Training loss  1.313256897511428
Training loss  1.2642685602466792
Training loss  1.2357862400199602
Training loss  1.266654408681798
Training loss  1.27263109393442
Training loss  1.5146723742522614
Training loss  1.391682121160225
Training loss  1.2714302491229617
Training loss  1.2826392648572849
Training loss  1.2383833009287224
Training loss  1.3292541983290136
Training loss  1.3245201135403253
Training loss  1.239190689894706
Training loss  1.3186842974494486
Training loss  1.2922192543335327
Training loss  1.3469662606265742
Training loss  1.2608628235660084
Training loss  2.904125645149887
Training loss  1.4002117085578558
Training loss  1.2366532737487752
Training loss  1.2786472047645072
Training loss  1.1300469217598987
Training loss  1.2679537860384678
Training loss  2.92210370519813
Training loss  2.8955284309021114
Training loss  1.3079757300589872
Training loss  1.2393724524456522
Training loss  1.288996731505102
Training loss  1.121352904834192
Training loss  1.2295915681633365
Training loss  1.2604668550372238
Training loss  1.2703189806207094
Training loss  2.938630837372282
Training loss  1.2314534999277247
Training loss  2.9120655509030873
Training loss  1.2751731287841048
Loss and Error 2.985260053041694 155.06575831542133
saving as ./model6//014_155.0658.w
Epoch Over
teacher_force 0.8199999999999998
Training loss  1.2272343635559082
Training loss  2.9043327906085814
Training loss  2.875762884794776
Training loss  1.231880489051973
Training loss  1.20893593195485
Training loss  1.240406102358264
Training loss  1.2814577579798756
Training loss  2.923939965956704
Training loss  1.1958589968474016
Training loss  1.2693707627735102
Training loss  1.1103624148129847
Training loss  1.3512722154397003
Training loss  1.274208338529789
Training loss  1.2706659363662323
Training loss  1.2272336546459126
Training loss  2.898766514795766
Training loss  1.2165524266993542
Training loss  1.3525698859980357
Training loss  1.218717391695452
Training loss  2.8897763240014265
Training loss  2.9130920490571475
Training loss  2.9278450352315213
Training loss  1.2440250103290265
Training loss  1.2195185882040187
Training loss  1.1792075425936426
Training loss  1.2536612054080034
Training loss  1.3591462556306306
Training loss  1.3023012962625526
Training loss  1.3350038781630253
Training loss  1.2873331285500884
Training loss  1.198901529587029
Training loss  1.2046889900601525
Training loss  2.9057984172200424
Training loss  2.912254378703381
Training loss  1.2911738670224804
Training loss  1.3325204174228675
Training loss  1.1925208793556186
Training loss  1.2868257014473303
Training loss  1.36163330078125
Training loss  1.2678969957652366
Training loss  1.3103213500976563
Training loss  1.1327463173004517
Training loss  1.2262354173242844
Training loss  1.1521625600961538
Training loss  1.3795827595735912
Training loss  1.2510521840573738
Training loss  1.1796005664665825
Training loss  2.9093700292908133
Training loss  1.344429178041372
Training loss  1.2640171574702563
Training loss  2.9084659976422635
Training loss  2.9060970408493003
Training loss  1.2394430326874253
Training loss  1.1359694908405173
Training loss  2.9095818014705883
Training loss  1.2559248798076923
Training loss  1.201668463226283
Training loss  2.9153978056923306
Training loss  2.907694897174035
Training loss  1.2584808016910307
Training loss  1.281981226279383
Training loss  1.2879968265376835
Training loss  0.9997551147915961
Training loss  1.274053344910903
Training loss  1.0769433385527718
Training loss  1.104538641150253
Training loss  2.906968018926184
Training loss  1.349148236358982
Training loss  1.2794974727091022
Training loss  1.17767311359424
Training loss  1.1987444559733074
Training loss  1.2575604563779357
Training loss  1.2371150703489502
Training loss  1.2990702985116382
Training loss  1.2461080363285777
Training loss  1.2306140816030775
Training loss  2.9124932332677167
Training loss  1.3164167041864536
Loss and Error 2.993406912854312 130.15137163466792
saving as ./model6//015_130.1514.w
Epoch Over
teacher_force 0.7599999999999998
Training loss  1.225218469586643
Training loss  1.191862128146453
Training loss  1.185823580973241
Training loss  1.2388064223642632
Training loss  2.961114214068526
Training loss  1.3181258555543054
Training loss  1.3691253039692501
Training loss  1.2711447646887812
Training loss  1.248939955754675
Training loss  1.336962505927196
Training loss  1.2383515422189664
Training loss  1.2048502429343448
Training loss  1.2421656074859888
Training loss  1.2326616003928255
Training loss  1.1608215154889552
Training loss  1.1196523872910031
Training loss  1.2730680541001893
Training loss  1.1954003734622145
Training loss  1.253486229037486
Training loss  1.302517017081484
Training loss  1.2815298685213414
Training loss  1.2363110299759397
Training loss  2.884049938860694
Training loss  1.208914567771369
Training loss  1.2181222293769274
Training loss  2.8816828421912115
Training loss  1.1927236074298906
Training loss  1.2483368532724648
Training loss  1.2262217040859225
Training loss  1.2697783995235024
Training loss  1.2442188968515038
Training loss  1.2723517792322356
Training loss  1.1938895540240229
Training loss  1.2954410772267906
Training loss  1.2207678955842585
Training loss  2.9004969219532555
Training loss  1.226904329933988
Training loss  2.90203504388339
Training loss  1.2424229677786411
Training loss  2.9013801775421224
Training loss  1.2728397492439516
Training loss  2.919803282474181
Training loss  2.8835992023928214
Training loss  1.2357967328001902
Training loss  1.2621214434899117
Training loss  1.2409527493590171
Training loss  1.1912493730570417
Training loss  1.2504477863690884
Training loss  1.2068106621234
Training loss  1.291115133883659
Training loss  1.1939160461903364
Training loss  1.1982111791567525
Training loss  1.3072672238479093
Training loss  1.3042678593705446
Training loss  2.9266797534653812
Training loss  1.1588712993421053
Training loss  1.2199037871258185
Training loss  1.263817137104862
Training loss  2.8989225404452434
Training loss  1.1656426413371037
Training loss  1.2198413907205659
Training loss  1.300323038856305
Training loss  1.15159484375
Training loss  1.1844380601276945
Training loss  1.156068158988614
Training loss  1.3655525103007278
Training loss  2.8993208779542097
Training loss  1.2705461120605468
Training loss  1.1369172942895684
Training loss  1.1941781953444748
Training loss  1.179698717886526
Training loss  1.2136898400648586
Training loss  1.2093400874789206
Training loss  2.894947804625077
Training loss  2.909188735729649
Training loss  1.314719375
Training loss  2.8985508352751053
Training loss  2.8977637900191797
Loss and Error 2.9802306055029795 127.40536995883275
saving as ./model6//016_127.4054.w
Epoch Over
teacher_force 0.7599999999999998
Training loss  1.310200899948814
Training loss  1.2385637232976945
Training loss  1.24830591931914
Training loss  1.1689492100195562
Training loss  1.2627504746022382
Training loss  1.2798812427080632
Training loss  1.2052471396130306
Training loss  1.2673095113413346
Training loss  1.241063819837047
Training loss  1.1833194650462289
Training loss  1.1489072389240507
Training loss  1.3027661354563433
Training loss  1.1846820108251634
Training loss  1.1237196252917638
Training loss  1.2142484935241389
Training loss  1.2241327646493128
Training loss  2.8975025807057055
Training loss  1.2142236972584948
Training loss  1.140592094428697
Training loss  2.917311099245782
Training loss  1.3338338006042025
Training loss  2.9082624217527386
Training loss  1.3623690497959655
Training loss  1.5005177674861496
Training loss  1.2144202279074763
Training loss  1.1523159708753556
Training loss  2.8688862025304966
Training loss  1.2220837606579793
Training loss  2.918125144364028
Training loss  1.1923823456919216
Training loss  1.2663255331605023
Training loss  1.2687779718907035
Training loss  1.227723135599276
Training loss  1.1996053116709717
Training loss  1.1224161847933072
Training loss  1.232456507908275
Training loss  1.151766631175025
Training loss  1.2555727166944364
Training loss  1.2464171111655609
Training loss  1.2333878091982444
Training loss  1.1372873482964667
Training loss  1.0673721150042599
Training loss  2.91337208478228
Training loss  1.2832673488398894
Training loss  2.89480893920068
Training loss  1.1660547699103376
Training loss  2.924028491797597
Training loss  1.1620562763584201
Training loss  1.1798619226431746
Training loss  1.2345815805288463
Training loss  1.1470771284955044
Training loss  1.2026747388998251
Training loss  1.2471597566282806
Training loss  1.2513607726240392
Training loss  1.1920400455298013
Training loss  1.2862050725125718
Training loss  1.1782613009692513
Training loss  1.1972991704318292
Training loss  1.1344896138008005
Training loss  1.3043506735049886
Training loss  1.0983130087717594
Training loss  1.1345011221830243
Training loss  1.1531728469451832
Training loss  2.903069019461643
Training loss  1.195307807426948
Training loss  1.1808273368428324
Training loss  1.161488518870549
Training loss  1.2340562390318808
Training loss  1.1745082998181706
Training loss  1.2377642802443165
Training loss  1.194768682090803
Training loss  1.3493578901264813
Training loss  1.2879682231666199
Training loss  1.1720229640151516
Training loss  1.1268798267985907
Training loss  1.2976959695962118
Training loss  1.1955282022664835
Training loss  1.1341039407603168
Loss and Error 2.9761349868860405 99.32966592589894
saving as ./model6//017_99.3297.w
Epoch Over
teacher_force 0.7599999999999998
Training loss  1.1906621758002287
Training loss  1.1326894903960834
Training loss  1.3484059343434343
Training loss  1.21122578125
Training loss  0.9860257470346715
Training loss  1.1605543581175857
Training loss  1.0570111205246275
Training loss  1.131743277616279
Training loss  1.2417367788461537
Training loss  1.2084578453160477
Training loss  1.1498777780473604
Training loss  1.2642223488365203
Training loss  1.2053584965396689
Training loss  1.2041193363683818
Training loss  1.0540846604567307
Training loss  1.1469496030688395
Training loss  1.201754593339557
Training loss  1.1993678554926068
Training loss  1.226451930068814
Training loss  2.883496581218802
Training loss  2.9256579996307734
Training loss  2.9144153897027065
Training loss  1.266856305040996
Training loss  1.167380070202808
Training loss  1.2594599138226426
Training loss  1.2587241434589225
Training loss  1.2102327686590706
Training loss  2.894519551304906
Training loss  1.144675556429856
Training loss  1.1778598613292404
Training loss  1.316836166788954
Training loss  1.2007076572189133
Training loss  1.179824344103377
Training loss  1.2161609046603716
Training loss  2.883884475702777
Training loss  1.1588076565842758
Training loss  1.1980645325027648
Training loss  1.1974601745605469
Training loss  1.279693165646444
Training loss  1.2402236343359985
Training loss  1.1324753275156685
Training loss  1.2616184753885533
Training loss  2.8874241240186103
Training loss  2.893052695200573
Training loss  1.2451213203822329
Training loss  1.1670795626540749
Training loss  1.1535119409796166
Training loss  1.1820061565109
Training loss  1.180107844113775
Training loss  1.1973176412156357
Training loss  2.8869488516830293
Training loss  2.914621226352546
Training loss  1.2430329046072621
Training loss  2.9044892520464303
Training loss  1.1028424475187195
Training loss  1.1135325800421778
Training loss  1.196649924017529
Training loss  2.904894606583939
Training loss  1.1873802443561547
Training loss  2.906680680013612
Training loss  1.1629870830138036
Training loss  1.106291795095894
Training loss  2.892305885590647
Training loss  1.094861668779938
Training loss  2.921560378959276
Training loss  1.0796887954681766
Training loss  2.91286083927532
Training loss  1.1234330738309926
Training loss  1.1670483383053718
Training loss  1.1966584135098473
Training loss  1.1689164456646584
Training loss  1.3325866641872848
Training loss  1.089386184237192
Training loss  1.2249698044298936
Training loss  1.3133276999364212
Training loss  1.15744203977947
Training loss  2.8949830056729264
Training loss  2.9111486144822005
Loss and Error 2.9867365821463445 91.30205107654196
saving as ./model6//018_91.3021.w
Epoch Over
teacher_force 0.7599999999999998
Training loss  1.078523681042404
Training loss  1.133008037255466
Training loss  1.2093218999608093
Training loss  1.07932505940803
Training loss  1.1679008979301948
Training loss  1.1040716379390882
Training loss  1.0518982699725419
Training loss  1.1871991298593552
Training loss  1.235274995497801
Training loss  2.8984468900240383
Training loss  1.177056257546284
Training loss  1.1813345828694684
Training loss  1.0844172106686
Training loss  1.2801093768356437
Training loss  1.0449522246277112
Training loss  2.8879361281150393
Training loss  2.908695338716443
Training loss  1.213775126790573
Training loss  1.203519579390405
Training loss  1.0685364510686741
Training loss  1.28974018943954
Training loss  1.1739006319768952
Training loss  1.10741568529866
Training loss  1.191376362911463
Training loss  2.9130817873488066
Training loss  1.1524010955122181
Training loss  1.112375042002688
Training loss  1.156919210843089
Training loss  2.88356902440816
Training loss  1.2307276897777983
Training loss  1.2133099084713104
Training loss  1.1495552409301926
Training loss  1.1634628642476212
Training loss  1.075661490483539
Training loss  1.0923243596724364
Training loss  1.2192937829533004
Training loss  1.121833108098735
Training loss  1.090571540076117
Training loss  1.054668589863429
Training loss  1.2428085711979642
Training loss  2.892194606374531
Training loss  1.2193224142772199
Training loss  1.1491082103782944
Training loss  1.1654474457484099
Training loss  1.255644559111983
Training loss  1.2292766933162154
Training loss  1.1860371591616612
Training loss  1.210690702274728
Training loss  1.0650530179033302
Training loss  1.1998728872585438
Training loss  1.1466093826507415
Training loss  1.2180959283394692
Training loss  1.0497003364847244
Training loss  1.158760417927154
Training loss  2.864989505597015
Training loss  1.1321613648197357
Training loss  2.8744766077277575
Training loss  2.9235669513081395
Training loss  1.1954516443955578
Training loss  1.1899389231924622
Training loss  1.1280302410558465
Training loss  2.90999696656683
Training loss  1.0525119855214133
Training loss  1.1877677622210518
Training loss  1.1723274981157235
Training loss  1.1040439741813202
Training loss  1.1905546607925832
Training loss  1.08991300353456
Training loss  2.8882280103299416
Training loss  1.1441116511682157
Training loss  1.2032123234437895
Training loss  1.1093649059414532
Training loss  1.275571683051904
Training loss  2.8831938166283444
Training loss  2.8780713891032836
Training loss  2.8909396837704144
Training loss  1.1765424757604226
Training loss  1.269595739976415
Loss and Error 2.9721465400222478 98.36697147437066
saving as ./model6//019_98.3670.w
Epoch Over
teacher_force 0.7599999999999998
Training loss  1.2067612650950668
Training loss  2.9257369059217213
Training loss  1.1293706408135054
Training loss  2.8714180828010947
Training loss  1.081547438496291
Training loss  1.1632095461907295
Training loss  2.9172244740521003
Training loss  2.869524690208559
Training loss  1.1687041859567902
Training loss  2.8847518705985915
Training loss  1.2487462425712375
Training loss  1.059485095507651
Training loss  1.093369048943833
Training loss  1.1790179966517857
Training loss  1.2348739684746304
Training loss  1.196489740548806
Training loss  1.2928846951188713
Training loss  2.870311703556305
Training loss  1.1589873727725026
Training loss  1.1249670898988298
Training loss  1.0414756306719966
Training loss  2.8553264429690275
Training loss  1.149292244473354
Training loss  1.021171062390126
Training loss  1.176319256051753
Training loss  1.066851203340621
Training loss  1.137064243684914
Training loss  2.9095051206391433
Training loss  1.2303517905778267
Training loss  1.126425156570936
Training loss  1.1952798624913168
Training loss  1.136606144383698
Training loss  2.90804900506708
Training loss  1.173417210616847
Training loss  1.0110902825936199
Training loss  1.0712304912773933
Training loss  2.9112621897163122
Training loss  1.0723200361858043
Training loss  1.0781923767309898
Training loss  2.902280745967742
Training loss  1.1768737753818554
Training loss  2.882947488473802
Training loss  1.0940916408746006
Training loss  2.8989658986292963
Training loss  2.8842498129986702
Training loss  1.1051403651421674
Training loss  1.0083517441021397
Training loss  1.1725771634898947
Training loss  1.0966207225420321
Training loss  1.1201544603188738
Training loss  1.1712838367474916
Training loss  1.1699284411292787
Training loss  1.0651302228377626
Training loss  1.0348355113876224
Training loss  1.1087514823984688
Training loss  2.9084652202973924
Training loss  1.1490456176901933
Training loss  1.1721449667084956
Training loss  2.881142700444218
Training loss  2.8770588514974373
Training loss  1.0658249403106868
Training loss  1.14065248979397
Training loss  1.1532833860529377
Training loss  2.8658301748761583
Training loss  1.1572853611434324
Training loss  1.2190811346247268
Training loss  1.2932697892836718
Training loss  2.898539524628458
Training loss  2.9035002529448537
Training loss  2.893963133420086
Training loss  1.1473280377272912
Training loss  1.1856616657223928
Training loss  1.0416229257019993
Training loss  0.9454060178178316
Training loss  1.2042154897417003
Training loss  1.0907301272199454
Training loss  1.2353397934641195
Training loss  1.2055605506013218
Loss and Error 2.9916841967619328 114.80928266967831
saving as ./model6//020_114.8093.w
Epoch Over
teacher_force 0.6999999999999997
Training loss  1.1576847831738768
Training loss  1.006830385439075
Training loss  0.9804073720628923
Training loss  1.2246273870907207
Training loss  1.1825808690599173
Training loss  1.241858883304196
Training loss  1.1024961262569348
Training loss  1.0209301641739303
Training loss  1.0917280896169075
Training loss  1.0605852280444739
Training loss  1.2346564398871527
Training loss  1.0995603239155252
Training loss  1.0159654863786034
Training loss  2.89544977846908
Training loss  1.0102133918588063
Training loss  1.2251700867167519
Training loss  2.889400460090984
Training loss  1.215150894657258
Training loss  2.8843444534952605
Training loss  2.8891370841369333
Training loss  1.1893753600502417
Training loss  2.893879220409712
Training loss  2.8933097131131635
Training loss  1.0671297436554223
Training loss  1.1564392328573174
Training loss  2.918746193309595
Training loss  1.1313295398078622
Training loss  2.8748288244798235
Training loss  1.1408964883528514
Training loss  1.1147956372496493
Training loss  1.1340842432162703
Training loss  2.87357834539377
Training loss  2.8884575965802988
Training loss  1.2699726405261644
Training loss  2.907533804706854
Training loss  1.1128217334775405
Training loss  2.8974739033267953
Training loss  1.1288115818219473
Training loss  1.054588673212757
Training loss  1.066588124711552
Training loss  1.112255785616503
Training loss  1.0292174461820973
Training loss  2.905989266010019
Training loss  1.1573831434342643
Training loss  1.0326120637492193
Training loss  1.0957213837208115
Training loss  1.2166167531448024
Training loss  2.9172532433380085
Training loss  1.156328105491046
Training loss  1.1285497260477673
Training loss  2.8736934351533168
Training loss  1.2094195301226551
Training loss  1.078673193059816
Training loss  0.9886133707212844
Training loss  1.142701672511001
Training loss  1.0892047929883302
Training loss  2.8901358351429605
Training loss  1.2038104036563255
Training loss  1.043765193515602
Training loss  1.0215101339350772
Training loss  1.2009909817383393
Training loss  1.2395993986061151
Training loss  1.1749023711123563
Training loss  1.2086508285566575
Training loss  1.1217841269912456
Training loss  1.0504839309083405
Training loss  1.0729261216286308
Training loss  1.1556513592479676
Training loss  2.8565261052887285
Training loss  2.875778610641892
Training loss  2.94431311039065
Training loss  0.9911701544657423
Training loss  2.8926896937370956
Training loss  2.8860069821041217
Training loss  1.1184871897977942
Training loss  2.8767391622729934
Training loss  1.1719773225687162
Training loss  1.2730463255977496
Loss and Error 2.964387772197685 88.27006448322344
saving as ./model6//021_88.2701.w
Epoch Over
teacher_force 0.6999999999999997
Training loss  2.8675544147381484
Training loss  1.057398822969781
Training loss  1.1907834363339551
Training loss  1.067919872353575
Training loss  2.874683003960556
Training loss  2.867673773133502
Training loss  1.0847266852522695
Training loss  1.2014641286612544
Training loss  1.1264002903642099
Training loss  1.01508662402543
Training loss  2.9173926644898462
Training loss  1.116904560645022
Training loss  1.114034233825293
Training loss  1.0845166000779112
Training loss  0.9984383610192691
Training loss  1.071714420080028
Training loss  2.89788588765799
Training loss  1.1775339837125494
Training loss  2.8798391505831713
Training loss  1.0976597766974994
Training loss  1.1239152554266467
Training loss  1.1759669961033679
Training loss  1.1952396665715372
Training loss  2.8725428618920974
Training loss  1.2595761912188563
Training loss  1.2421631847183319
Training loss  1.1125977807718361
Training loss  1.1410386011698272
Training loss  1.0594100369751909
Training loss  2.90723919409313
Training loss  1.2096870476909776
Training loss  1.2082061347688497
Training loss  2.896543964879326
Training loss  2.8960852478839176
Training loss  2.8901548344477996
Training loss  1.1858281122230896
Training loss  2.877452407387348
Training loss  1.0405089851719616
Training loss  1.0827234229262994
Training loss  2.8923326284346684
Training loss  2.900449029793316
Training loss  1.1704082626385042
Training loss  2.8808201291617777
Training loss  1.0709860441758123
Training loss  2.8735230643794223
Training loss  2.8876527940099077
Training loss  1.2015908366392747
Training loss  1.2228307233780216
Training loss  1.1998211511240284
Training loss  1.0861847809321716
Training loss  1.1545220705984627
Training loss  2.888778870705664
Training loss  2.9092117884909765
Training loss  1.1168497679534943
Training loss  2.877271631954693
Training loss  2.8788888548717493
Training loss  2.8914073884795983
Training loss  1.0879547705838037
Training loss  1.1955397445436509
Training loss  2.8865509969325154
Training loss  1.1925270164405906
Training loss  1.1651741293532338
Training loss  0.9951126472012642
Training loss  1.1593519330246436
Training loss  1.2407797112665757
Training loss  1.2103550238778327
Training loss  1.11779225929283
Training loss  1.0190172333474277
Training loss  2.97345273714539
Training loss  2.8695686908577533
Training loss  2.8918088845064727
Training loss  2.870583221713635
Training loss  2.887829751170154
Training loss  1.014557091346154
Training loss  2.8957729901856575
Training loss  0.9179114428196042
Training loss  1.138498022623087
Training loss  1.1430160167605379
Loss and Error 2.9888198285986336 118.30485627891727
saving as ./model6//022_118.3049.w
Epoch Over
teacher_force 0.6999999999999997
Training loss  1.074743994878392
Training loss  2.894996780903328
Training loss  1.190878689931871
Training loss  2.879466053287237
Training loss  1.1235128489847717
Training loss  1.0917813246417198
Training loss  1.2116043703752644
Training loss  2.8901115866268383
Training loss  1.1489494243421052
Training loss  1.173333106977787
Training loss  1.0363755794046967
Training loss  2.8915131833192804
Training loss  2.866724933055638
Training loss  1.0792681797985781
Training loss  1.1126893316084738
Training loss  2.866041674152055
Training loss  1.0799412792084984
Training loss  2.8887843586387434
Training loss  1.0612454084834613
Training loss  1.1875370923490038
Training loss  1.2383907669824912
Training loss  1.125846703322676
Training loss  1.0929169280337592
Training loss  1.0539444449451976
Training loss  2.8800431530785393
Training loss  1.0960106452814902
Training loss  1.0783057466221317
Training loss  1.2128272262955582
Training loss  1.0491821323633532
Training loss  1.206159685553909
Training loss  1.1047706175386236
Training loss  1.2502677620740918
Training loss  1.1660493577288429
Training loss  1.0889003866861136
Training loss  1.0585279994376358
Training loss  1.0410448791442275
Training loss  0.9734704471328883
Training loss  1.1334447030141843
Training loss  1.062308458011583
Training loss  1.0153864246681914
Training loss  1.1223769291489614
Training loss  0.9614284905787608
Training loss  1.0780607917622325
Training loss  2.878679862298011
Training loss  2.8619683127418876
Training loss  1.0660347714880394
Training loss  2.894844994606704
Training loss  1.1001464995437247
Training loss  2.8601419588414636
Training loss  1.0555623879753742
Training loss  1.142782992356115
Training loss  1.0715492340455572
Training loss  1.0477618100331292
Training loss  2.875870589371566
Training loss  1.040925780105479
Training loss  1.120690752965147
Training loss  2.8717603782995704
Training loss  1.1462816167017862
Training loss  1.0751462032784456
Training loss  2.8833930773974714
Training loss  2.863232664800995
Training loss  1.1797412964310787
Training loss  1.1168521532239988
Training loss  2.890401281514136
Training loss  1.0463635312715072
Training loss  1.1265397091248304
Training loss  1.0960646577937874
Training loss  2.8866644309250304
Training loss  0.966654812578102
Training loss  2.8719310037851606
Training loss  1.1220018390025042
Training loss  1.0331036360589811
Training loss  2.8860573120668875
Training loss  2.8876334833044233
Training loss  1.2031427076285521
Training loss  1.052063913850143
Training loss  1.0091882375725212
Training loss  1.1833484885565757
Loss and Error 3.0087827633751383 82.80993843127254
saving as ./model6//023_82.8099.w
Epoch Over
teacher_force 0.6999999999999997
Training loss  1.0810177790040272
Training loss  2.9010331080071707
Training loss  2.866324068689976
Training loss  2.8799975075875213
Training loss  1.0468502188087405
Training loss  0.9728624098183728
Training loss  1.1781047092321886
Training loss  1.0639023168995037
Training loss  1.1162853803790984
Training loss  2.8980810922889164
Training loss  1.0646978840504504
Training loss  2.8802985053093257
Training loss  1.1071094936794705
Training loss  1.0199167720894344
Training loss  1.0612163386165494
Training loss  1.0739146706586826
Training loss  2.9119049444320546
Training loss  1.233644392778016
Training loss  2.888991080236764
Training loss  0.9533168247767857
Training loss  2.868830160830619
Training loss  1.0293946397268119
Training loss  1.154330494364754
Training loss  1.0550709360921031
Training loss  2.892971553814142
Training loss  1.0301421413432388
Training loss  1.1086075240194135
Training loss  1.0793388139317137
Training loss  1.1032831061940833
Training loss  2.8682625201494725
Training loss  1.0227135720056655
Training loss  1.1899227949309148
Training loss  1.0852871142184595
Training loss  1.1150307865466103
Training loss  1.0191338209515437
Training loss  1.076190229058652
Training loss  1.039958860473151
Training loss  1.1220341380049543
Training loss  1.0853718258836333
Training loss  2.8870289610639936
Training loss  1.0349393385356158
Training loss  1.1844216987781955
Training loss  1.0590220853365384
Training loss  1.2020986065857031
Training loss  2.8776514414764867
Training loss  1.1324569755939704
Training loss  2.849464665365797
Training loss  1.0821226783179236
Training loss  1.1047840620623248
Training loss  1.000169089988426
Training loss  2.8389442924347157
Training loss  2.8679874727668846
Training loss  1.010376911569149
Training loss  2.866582465542187
Training loss  1.0571061659314434
Training loss  1.0743561060130071
Training loss  1.1434178832934143
Training loss  0.9360613029233871
Training loss  1.0136129852265399
Training loss  0.9457930106162715
Training loss  2.882951750578704
Training loss  2.8717474819214877
Training loss  2.872831444229647
Training loss  1.1443638350414511
Training loss  2.878749016576405
Training loss  1.1053311471415264
Training loss  1.1214755452085323
Training loss  1.0329761263242345
Training loss  1.1899020847651196
Training loss  2.8380362775233046
Training loss  2.827464169539591
Training loss  2.8676083838061843
Training loss  1.0249525300982267
Training loss  1.0199726688305153
Training loss  1.043270680874209
Training loss  1.0265742425091624
Training loss  1.1717471769059744
Training loss  1.0850387216788628
Loss and Error 2.951628053694479 90.2082043061678
saving as ./model6//024_90.2082.w
Epoch Over
teacher_force 0.6999999999999997
Training loss  2.8537788205030488
Training loss  1.0618692690767284
Training loss  1.071905717441357
Training loss  2.880773490352091
Training loss  1.0289476441885126
Training loss  1.0954054222212604
Training loss  1.1033582348508884
Training loss  2.9034957598341062
Training loss  1.0061381784918562
Training loss  1.0697052470144701
Training loss  2.894296691930287
Training loss  0.9894869773092011
Training loss  1.2283166201869613
Training loss  2.901910948654075
Training loss  1.1648917250425674
Training loss  0.9801481725003548
Training loss  1.017858343794835
Training loss  0.9675848538077969
Training loss  2.892549247405459
Training loss  1.019741221451092
Training loss  0.9907907361635533
Training loss  1.2420857084690553
Training loss  2.897801185186545
Training loss  0.9510942406315809
Training loss  0.9732848193162427
Training loss  2.8964465597401494
Training loss  1.1313935233353152
Training loss  2.8895075667706074
Training loss  1.116902430118248
Training loss  2.849237956221634
Training loss  2.85444237470344
Training loss  1.1169910371783942
Training loss  0.938230243992529
Training loss  1.1747323055893073
Training loss  1.1412669838177667
Training loss  1.0885532924107142
Training loss  1.1046314997913576
Training loss  1.0146598066128667
Training loss  1.1471784416347492
Training loss  1.076187198793844
Training loss  2.9117634676303523
Training loss  2.9367424919937313
Training loss  1.1717945050993723
Training loss  2.895621608194803
Training loss  2.8592142263302036
Training loss  0.9853663457257347
Training loss  0.9897400583170033
Training loss  2.8741087817872493
Training loss  1.1326220451865352
Training loss  1.0421907123766447
Training loss  1.0898923481652276
Training loss  2.8556927747110255
Training loss  1.0809889006840707
Training loss  1.0143811227044286
Training loss  0.98919372145636
Training loss  2.8667220386559498
Training loss  1.126288052262931
Training loss  1.0446463429077766
Training loss  1.0199862393465908
Training loss  0.9694135130507849
Training loss  2.8592052648336828
Training loss  1.163100114326488
Training loss  0.9842767588873826
Training loss  0.9106767162434483
Training loss  1.0146553540819017
Training loss  2.8847436203116086
Training loss  0.978451076751265
Training loss  2.88103732365227
Training loss  0.996410709530194
Training loss  1.0703760236733582
Training loss  2.899230399632563
Training loss  0.930753783768122
Training loss  2.887040205210935
Training loss  1.1524670862268518
Training loss  2.8811526805850356
Training loss  1.0258630842314738
Training loss  1.15516877497514
Training loss  1.1519765832974476
Loss and Error 2.9537060738794927 90.80658676090204
saving as ./model6//025_90.8066.w
Epoch Over
teacher_force 0.6999999999999997
Training loss  1.1401899605424304
Training loss  1.0549319577937235
Training loss  2.8493071404519603
Training loss  2.866365513392857
Training loss  1.0247465093085106
Training loss  0.984471787262197
Training loss  1.1264407143600776
Training loss  1.053954170798492
Training loss  1.089537319566203
Training loss  0.9574426992369428
Training loss  2.8835926275143677
Training loss  0.954301455287348
Training loss  0.9941906710283472
Training loss  1.0804795875699273
Training loss  1.1021400369623655
Training loss  1.1777835386633553
Training loss  1.1284335642603078
Training loss  1.039061874799936
Training loss  1.0593667386418435
Training loss  1.0654700969827586
Training loss  1.1309435565924413
Training loss  2.880570659521921
Training loss  2.879313842658168
Training loss  1.1420053976008602
Training loss  1.0263177054268973
Training loss  2.8660606564108644
Training loss  1.1048670115029045
Training loss  1.125542314690759
Training loss  1.0199088959989009
Training loss  1.2119935726950355
Training loss  1.0356196095226844
Training loss  1.1026498367537314
Training loss  1.0379688194033165
Training loss  1.0741385323660715
Training loss  1.0522539037507996
Training loss  1.1409914608620941
Training loss  1.1301486802645728
Training loss  1.060424076220162
Training loss  1.1856506052967908
Training loss  1.065507417173824
Training loss  1.0298081698158914
Training loss  1.0171082347575329
Training loss  1.148275994103971
Training loss  0.9519505299388112
Training loss  1.0136021003579694
Training loss  1.002356115263397
Training loss  1.084864236149117
Training loss  2.8807888836927815
Training loss  1.0645697939712389
Training loss  2.8786684200794146
Training loss  0.9787374015982825
Training loss  2.857716216326132
Training loss  0.9869543922025008
Training loss  1.1557981502679777
Training loss  1.0314826988549908
Training loss  1.0708439665020637
Training loss  2.8611045575271246
Training loss  2.8479234022929227
Training loss  2.881098731354197
Training loss  2.885562106053943
Training loss  1.0177410199294077
Training loss  1.10088046954828
Training loss  1.0492560981583072
Training loss  2.859259958223031
Training loss  1.0942982397111012
Training loss  1.010918293778366
Training loss  1.0850003844734253
Training loss  1.0540906687609093
Training loss  1.1297548597440945
Training loss  1.1362283007868448
Training loss  0.879688416800678
Training loss  2.848310691362054
Training loss  1.0477398556188562
Training loss  2.878749907057274
Training loss  1.052096953451883
Training loss  1.051048040665643
Training loss  1.0079374642959154
Training loss  1.0782778173460996
Loss and Error 2.9507225387760307 83.51488214506904
saving as ./model6//026_83.5149.w
Epoch Over
teacher_force 0.6999999999999997
Training loss  1.1702069575688918
Training loss  1.066273790724734
Training loss  1.1249150879200573
Training loss  2.9022257952213426
Training loss  1.1093723274546967
Training loss  1.1007042214736606
Training loss  0.9477289001203638
Training loss  1.1222798314037152
Training loss  1.0501769454698557
Training loss  2.894505442942943
Training loss  1.0023430036884375
Training loss  1.1977956974708936
Training loss  2.9031229987191804
Training loss  0.9946033288793783
Training loss  1.1725543834028413
Training loss  1.0182306589373222
Training loss  2.8536059596649137
Training loss  1.08202443159554
Training loss  1.1079490794141593
Training loss  1.0943308863146552
Training loss  1.0619370351693944
Training loss  2.8487639925373136
Training loss  1.046644769712936
Training loss  2.862780560077827
Training loss  1.074165542737986
Training loss  2.8821461886993602
Training loss  1.0644432777837798
Training loss  1.1270601320170264
Training loss  1.0989733227142666
Training loss  2.8590082484499555
Training loss  0.9622723544524241
Training loss  1.1016479845105205
Training loss  2.889806340876218
Training loss  1.1240655034824316
Training loss  1.131076491836654
Training loss  1.0132044538666931
Training loss  1.0067387734347846
Training loss  1.0551660221310792
Training loss  2.869533516682785
Training loss  1.065751209816991
Training loss  1.1432102009401484
Training loss  1.0593337095500341
Training loss  2.8598275385778535
Training loss  1.0830962604683623
Training loss  1.0464735778348084
Training loss  0.9934805907365305
Training loss  2.8897708704694
Training loss  1.0655701414323409
Training loss  2.8955551228647436
Training loss  2.8802153038008567
Training loss  1.0735075649581634
Training loss  2.861463863416988
Training loss  2.874704123301532
Training loss  1.0115747049870918
Training loss  1.1728806217476546
Training loss  0.9662614286711334
Training loss  1.032781942281066
Training loss  1.0219791844709054
Training loss  2.8878848736102762
Training loss  1.1099042880086738
Training loss  1.0239060924899193
Training loss  0.9779528468015786
Training loss  0.9318448153409091
Training loss  2.8725710611294684
Training loss  0.9568990133453648
Training loss  1.0178323906158733
Training loss  1.0725432398636685
Training loss  2.8701041342866023
Training loss  1.0562221270543344
Training loss  2.869041836503623
Training loss  1.052537830165143
Training loss  1.1274378283270474
Training loss  2.902246673765239
Training loss  1.017753135711038
Training loss  1.0932829088042466
Training loss  1.0195396035975441
Training loss  1.1044230008290623
Training loss  1.058389258625818
Loss and Error 2.9661517534585182 90.1708623264964
saving as ./model6//027_90.1709.w
Epoch Over
teacher_force 0.6999999999999997
Training loss  2.8738672234801963
Training loss  0.9954560002873175
Training loss  1.0200151691617811
Training loss  0.9917099110401459
Training loss  2.85172625765882
Training loss  1.1195342084798485
Training loss  0.9783039634715994
Training loss  1.0838027130552608
Training loss  0.8927645480312705
Training loss  1.0050869511540061
Training loss  0.9948512731481481
Training loss  0.8718688363621854
Training loss  0.9854449389940239
Training loss  1.0174647688935416
Training loss  1.0426938531711556
Training loss  0.9038646475616592
Training loss  1.0283080799843425
Training loss  0.8891213654366306
Training loss  1.0911196745074465
Training loss  2.8596118818074774
Training loss  0.95527671168914
Training loss  2.912862878655989
Training loss  1.0300928766993958
Training loss  1.0411364142147677
Training loss  1.2005975029108449
Training loss  1.0094492422479233
Training loss  0.8982379896069005
Training loss  1.0646018208182828
Training loss  1.0531498783120106
Training loss  1.0476472320645822
Training loss  1.0748151985036523
Training loss  2.872090426145097
Training loss  2.8729899187424057
Training loss  0.9813029199437711
Training loss  2.8646089037365954
Training loss  1.0318248968927388
Training loss  2.904788398848229
Training loss  1.0688060015644494
Training loss  2.8761001192641817
Training loss  0.8907762433244326
Training loss  1.1386815298318476
Training loss  2.885046893001152
Training loss  2.868130060993472
Training loss  0.9879592097355769
Training loss  1.0209280731442911
Training loss  1.1881528017394707
Training loss  2.878708378275807
Training loss  1.1810206011996243
Training loss  1.049019230222239
Training loss  2.8690388655462185
Training loss  1.089850300287985
Training loss  1.0873790425792378
Training loss  0.9921943731671554
Training loss  1.0339455685150052
Training loss  0.992040974116251
Training loss  1.097259641295394
Training loss  1.0735983631131583
Training loss  1.062036445448923
Training loss  2.8710793096730938
Training loss  2.863027981268546
Training loss  2.875299569945438
Training loss  1.0391838304924241
Training loss  1.1694724762248792
Training loss  2.86344068877551
Training loss  1.0473151033784807
Training loss  1.0762624284182147
Training loss  1.0636567253095277
Training loss  1.0271599264705882
Training loss  2.877815867659207
Training loss  1.0334430773110963
Training loss  0.9794788120375473
Training loss  0.9183410371646796
Training loss  1.0251221262164705
Training loss  1.1562989899536757
Training loss  1.0344458523272553
Training loss  2.9498083191095295
Training loss  2.8526051806857913
Training loss  1.1291332714782059
Loss and Error 2.965357081930842 101.19949724944442
saving as ./model6//028_101.1995.w
Epoch Over
teacher_force 0.6999999999999997
Training loss  1.059454350879983
Training loss  0.9264258705867638
Training loss  1.1276647444403969
Training loss  2.834149153985803
Training loss  0.9789368449288978
Training loss  0.9748511098899932
Training loss  1.1016675964567642
Training loss  2.8727482091558634
Training loss  1.03044861066481
Training loss  2.86427449569097
Training loss  1.177624321291689
Training loss  2.8667208455123796
Training loss  2.8676452476353127
Training loss  1.1280638872716278
Training loss  1.0391818977100369
Training loss  2.87388101623659
Training loss  2.8789727850274724
Training loss  2.864975909019975
Training loss  2.8758519629225736
Training loss  1.0872179255090335
Training loss  0.918095007036609
Training loss  1.0310515639062934
Training loss  2.880025445292621
Training loss  1.1044176966214727
Training loss  1.0395709749702469
Training loss  2.8356724920662906
Training loss  1.0970611931295955
Training loss  1.0587670829319362
Training loss  0.9416694253177966
Training loss  2.8778541185099575
Training loss  2.8674275488826817
Training loss  1.0440221394416538
Training loss  2.8746290849390075
Training loss  1.0221225699714884
Training loss  0.9585972256928113
Training loss  1.0430886088937954
Training loss  0.9344138198757764
Training loss  2.878973966072313
Training loss  1.0893178796812326
Training loss  2.8943088167192648
Training loss  1.0085495252543004
Training loss  0.9484283248030798
Training loss  1.003796747579001
Training loss  1.0822169002833388
Training loss  1.1303891810436686
Training loss  2.853362766723272
Training loss  1.0953795352224576
Training loss  2.847711454211419
Training loss  0.9827152946693601
Training loss  1.060451029101203
Training loss  1.06544189453125
Training loss  1.0478005327247326
Training loss  1.0062500882967902
Training loss  0.9072265625
Training loss  1.0283498477294835
Training loss  2.861935334158416
Training loss  2.8762018785216488
Training loss  2.855068519467213
Training loss  2.912191642992424
Training loss  2.867818459057768
Training loss  2.8640878519717696
Training loss  1.0542664555170593
Training loss  0.9390903365946262
Training loss  2.8531570817956973
Training loss  1.037116272046818
Training loss  1.0102594729743486
Training loss  1.0762552950816293
Training loss  0.9903425549241032
Training loss  2.8548755960874135
Training loss  2.8666843259549895
Training loss  1.023195832790152
Training loss  0.8741113214544911
Training loss  0.9736404604833797
Training loss  0.9833384540024729
Training loss  1.0686927560366362
Training loss  1.0685569036024962
Training loss  0.9962662675340983
Training loss  2.867420911713563
Loss and Error 2.9406777835503797 88.84476665816605
saving as ./model6//029_88.8448.w
Epoch Over
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     