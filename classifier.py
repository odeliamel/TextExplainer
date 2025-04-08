import re

from typing import Callable
from typing import Union

import torch
import torch.nn as nn
from torch.autograd import Variable


class Classifier():
    def __init__(self,
                 text_to_tokens: Callable[[str], torch.LongTensor],
                 embedding: Callable[[torch.Tensor], torch.Tensor],
                 model: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]) -> None:
        """
        __init__ creates new classifier obj

        Args:
            text_to_tokens (Callable[[str], torch.LongTensor]): function gets textual input and returns continuous tokens (incl. split, vocab, etc.)
            embedding (Callable[[torch.Tensor], torch.Tensor]): function gets tekenized input and returns embedded tokens (i.g., linear torch Embedding, conv embedding, etc.)
            classification Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]: function gets the embedded tokens and performs the classification (i.g., using linear, deep, LSTM, conv, etc.), backwards will be used for this function.

        Need to satisfy res = classification(embedding(text_to_tokens(data))) while (res in [0,1])
        """
        self.text_to_tokens = text_to_tokens
        self.embedding = embedding
        self.model = model

    def classify(self, x: str):
        out = self.model(self.embedding(self.text_to_tokens(x)))
        return torch.round(out)

    def get_gradient(self, data, label):
        input_embedded = self.embedding(self.text_to_tokens(data))
        input_embedded = Variable(input_embedded, requires_grad = True)
        with torch.enable_grad():
            self.model.zero_grad()
            out = self.model(input_embedded)
            label = label.detach()
            loss = torch.sum(torch.abs(out-label))
            
            loss.backward()
            grad = input_embedded.grad.data.detach()
        return grad
       