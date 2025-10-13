import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
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
    text = "Frogs are amazing amphibians that thrive in both aquatic and terrestrial environments. They begin their life as eggs in water, which hatch into tadpoles. As tadpoles grow, they develop legs and eventually jump onto land. Adult frogs are known for their distinctive croaks and play a vital role in ecosystems by controlling insect populations. Frogs have smooth, moist skin that helps them absorb water and breathe. They are found all over the world, from tropical rainforests to temperate regions. Some species of frogs are brightly colored, which can serve as a warning to predators about their toxicity. Frogs are also important indicators of environmental health, as they are sensitive to changes in their habitats."
    tokens, LSTM_dict = tokenize(text, LSTM_dict)
    model = LSTMModel(vocab_size=len(LSTM_dict), embedding_dim=500, hidden_dim=1000)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5):
        for i in range(1, len(tokens) - 2):
            text_tokens = np.array([])
            for word in tokens[:-i]: 
                text_tokens = np.append(text_tokens, word)
            last_word = torch.tensor([tokens[-i]], dtype=torch.long) 
            input_sequence = torch.tensor(text_tokens, dtype=torch.long).unsqueeze(1)
            output = model(input_sequence)
            loss = criterion(output, last_word)  # предсказываем следующее слово
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()