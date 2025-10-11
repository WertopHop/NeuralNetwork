import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class LTSM(nn.Module):
    def __init__(self):
        self.LSTM_dict = {}
        text = "This is a sample text for building the LSTM dictionary This text is for testing the LSTM model"
        tokens = self.tokenize(text)
        print("Tokens:", tokens)

    def tokenize(self, text):
        text_tokens = np.array([])
        text_words = np.array(text.split())
        for word in text_words:
            if word not in self.LSTM_dict:
                self.LSTM_dict[word] = len(self.LSTM_dict)
            text_tokens = np.append(text_tokens, self.LSTM_dict[word])
        return text_tokens