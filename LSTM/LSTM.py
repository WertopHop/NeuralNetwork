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


def load_and_tokenize(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    paragraphs = text.split('@')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    word_to_token = {}
    batches = []
    for paragraph in paragraphs:
        words = paragraph.split()
        tokens = []
        for word in words:
            if word not in word_to_token:
                word_to_token[word] = len(word_to_token)
            tokens.append(word_to_token[word])
        if tokens:
            batches.append(np.array(tokens))
    
    return batches, word_to_token

    
def learning(model, batches, criterion, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        total_samples = 0
        for batch_idx, tokens in enumerate(batches):
            print(f"Processing batch {batch_idx + 1}/{len(batches)}", end='\r')
            if len(tokens) < 2:
                continue
            for i in range(1, len(tokens)):
                input_tokens = tokens[:i]
                target_token = tokens[i]
                input_sequence = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(1).to(device)
                target = torch.tensor([target_token], dtype=torch.long).to(device)
                output = model(input_sequence)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                total_samples += 1
        avg_loss = epoch_loss / total_samples if total_samples > 0 else 0
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

def evaluate(model, count, words, word_to_token, device):
    model.eval()
    with torch.no_grad():
        for _ in range(count):
            input_tokens = [word_to_token.get(word, 0) for word in words]
            input_sequence = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(1).to(device)
            output = model(input_sequence)
            predicted_token = torch.argmax(output, dim=1).item()
            for word, token in word_to_token.items():
                if token == predicted_token:
                    words.append(word)
                    print(word, end=' ')
                    break

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")    
    batches, word_to_token = load_and_tokenize('text.txt')
    vocab_size = len(word_to_token)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of batches: {len(batches)}")
    model = LSTMModel(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    learning(model, batches, criterion, optimizer, device, epochs=5)
    
    input_text = input("Write the beginning and the number of words to generate: ").split()
    if len(input_text) < 2:
        print("Invalid input. Please provide a beginning text and a word count.")
        return
    begin_text = input_text[:-1]
    word_count = int(input_text[-1])
    print(' '.join(begin_text), end=' ')
    evaluate(model, word_count, begin_text, word_to_token, device)
    print()  


if __name__ == "__main__":
    main()