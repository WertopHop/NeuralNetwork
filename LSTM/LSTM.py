import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding=embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input):
        embedded = self.embedding(input)
        output, (h_n, c_n) = self.lstm(embedded)
        output = self.fc(output[-1])  
        return output

def tokenize(text, LSTM_dict):
    text_tokens = np.array([])
    text_words = np.array(text.split())
    for word in text_words:
        if word not in LSTM_dict:
            LSTM_dict[word] = len(LSTM_dict)
        text_tokens = np.append(text_tokens, LSTM_dict[word])
    return text_tokens, LSTM_dict
    
def main():
    LSTM_dict = {}
    text = "This is a sample text for building the LSTM dictionary This text is for testing the LSTM model"
    model = LSTMModel(vocab_size=4, embedding_dim=50, hidden_dim=100)
    tokens, LSTM_dict = tokenize(text, LSTM_dict)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    input_sequence = torch.tensor([LSTM_dict[0], LSTM_dict[1]])  
    output = model(input_sequence)
    loss = criterion(output, torch.tensor([LSTM_dict[2]]))  # предсказываем следующее слово
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Потеря: {loss.item():.4f}")


if __name__ == "__main__":
    main()