import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
from collections import Counter
import string
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

class SentimentRNN(nn.Module):
    def __init__(self,no_layers, vocab, vocab_size,hidden_dim,embedding_dim,output_dim=1,drop_prob=0.5):
        super(SentimentRNN,self).__init__()
 
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
 
        self.no_layers = no_layers
        self.vocab = vocab
        self.vocab_size = vocab_size
    
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        #lstm
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim,
                           num_layers=no_layers, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
    
        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sig = nn.Sigmoid()

    def tokenize(self, texts):
        word_seqs = np.array([np.array([self.vocab[preprocess_string(word)] for word in text.split() 
                         if preprocess_string(word) in self.vocab.keys()]) for text in texts])

        pad =  torch.tensor(padding_(word_seqs,500))
        ret = pad.to(device)    
        return ret    

        
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
        hidden = (h0.data,c0.data)
        return hidden
        
    def forward(self,embeds,hidden=None):
        # print("embeds shape", embeds.shape)
        batch_size = embeds.size(0)
        if hidden==None:
            hidden = self.init_hidden(batch_size)
        # batch_size = x.size(0)
        # embeddings and lstm_out
        # embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True
        #print(embeds.shape)  #[50, 500, 1000]
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) 
        
        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        # print(sig_out.shape)

        sig_out = sig_out[:, -1] # get last batch of labels
        # print(sig_out.shape)
        # out = torch.cat([1-sig_out.unsqueeze(1), sig_out.unsqueeze(1)], dim=1)

        # return last sigmoid output and hidden state
        return sig_out
            


import re
import numpy as np
import torch
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
from collections import Counter

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s

def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features


def token_split(texts):
     return texts.split()


# def predict_text(model, text):
#         inputs = tokenizer(text)
#         batch_size = 1
#         h = model.init_hidden(batch_size)
#         # h = tuple([each.data for each in h])
#         output, h = model(inputs, h)
#         return round(output.item()), h


# def predict_text_unrounded(model, text):
#         inputs = tokenizer(text)
#         batch_size = len(text)
#         # print("batch size", batch_size, inputs.shape)
#         h = model.init_hidden(batch_size)
#         h = tuple([each.data for each in h])
#         # print(h[0])
#         output, h = model(inputs, h)
#         return output, h

# def predict_text_unrounded2(model, text):
#         inputs = tokenizer(text)
#         batch_size = len(text)
#         # print("batch size", batch_size, inputs.shape)
#         h = model.init_hidden(batch_size)
#         h = tuple([each.data for each in h])
#         # print(h.shape)
#         output, h = model(inputs, h)
#         out = torch.cat([1-output.unsqueeze(1), output.unsqueeze(1)], dim=1)
#         # print(out.shape)
#         return out

        
