import torch
import torch.nn as nn
import torch.optim as optim
import openml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class OpenMLDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values if isinstance(X, pd.DataFrame) else X)
        if isinstance(y, pd.Series):
            if y.dtype.name == 'category':
                y = y.cat.codes.values
            else:
                y = y.values
        self.y = torch.LongTensor(y) if len(y.shape) == 1 else torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_openml_data(dataset_id, target_column=None):
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    if y is None:
        raise ValueError(f"Целевая переменная не найдена. Доступные столбцы: {list(X.columns)}")
    if isinstance(y, pd.Series) and y.dtype.name == 'category':
        y = y.cat.codes.values
    elif isinstance(y, pd.Series):
        y = y.values
    for col in X.columns:
        if X[col].dtype.name == 'category':
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else X[col].iloc[0])
            X[col] = X[col].cat.codes
        else:
            X[col] = X[col].fillna(X[col].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42
    )
    train_dataset = OpenMLDataset(X_train, y_train)
    test_dataset = OpenMLDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    input_size = X.shape[1]
    if len(y.shape) == 1:
        output_size = len(np.unique(y))
    else:
        output_size = y.shape[1]
    return train_loader, test_loader, input_size, output_size


class NeuralNetwork(nn.Module):
    def __init__(self, layers_size = torch.tensor([784, 256, 16, 10]), dropout_rate=0.2):
        super(NeuralNetwork, self).__init__()
        layers = []
        for i in range(len(layers_size) - 2):
            layers.append(nn.Linear(layers_size[i], layers_size[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(layers_size[-2], layers_size[-1]))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        loss_epoch = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
        print(f"Эпоха {epoch+1}/{epochs}, Потери: {loss_epoch/len(train_loader):.4f}")

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            if outputs.shape[1] > 1:
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
    avg_test_loss = test_loss / len(test_loader)
    if total > 0:  
        accuracy = 100 * correct / total
        print(f'Точность модели: {accuracy:.2f}%')
    print(f'Средние потери на тестовой выборке: {avg_test_loss:.4f}')
    return avg_test_loss


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    dataset_id = 554
    train_loader, test_loader, input_size, output_size = prepare_openml_data(dataset_id, target_column=None)
    layers_size = torch.tensor([input_size, 256, 16, output_size])
    model = NeuralNetwork(layers_size=layers_size).to(device) 
    Loss = nn.CrossEntropyLoss() if layers_size[-1] > 1 else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, Loss, optimizer, device, epochs=10)

    evaluate_model(model, test_loader, Loss, device)

    torch.save(model.state_dict(), "pytorch_model.pth")
    print("Модель сохранена в 'pytorch_model.pth'")

if __name__ == "__main__":
    main()