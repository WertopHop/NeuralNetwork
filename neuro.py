import numpy as np
import openml
import pandas as pd
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx) 

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

class NeuralNetwork():
    def __init__(self, photo = None, layers_size = np.array([784, 256, 32, 10])):
        self.layers_size = layers_size
        self.weights = []
        for i in range(len(layers_size) - 1):
            weight_matrix = np.random.normal(size=(layers_size[i+1], layers_size[i]))
            self.weights.append(weight_matrix)
        self.biases = []
        for i in range(1, len(layers_size)):
            bias_vector = np.zeros((layers_size[i], 1))
            self.biases.append(bias_vector)

        if photo is not None:
            self.feedforward(photo)
        

    def train(self, data, all_y_trues, learning_rate=0.001, epochs=10):
        len_layers = len(self.layers_size) - 1
        d_ypred_d_w = [np.array([]) for _ in range(len_layers)]
        d_ypred_d_b = [np.array([]) for _ in range(len_layers)]
        input_N = [np.array([]) for _ in range(len_layers)]
        sigmoid_N = [np.array([]) for _ in range(len_layers)]
        if os.path.exists('neural_network_weights.npz'):
            os.remove('neural_network_weights.npz')
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                x = np.array(x).reshape(-1, 1)
                y_true_temp = np.eye(10)[int(y_true)]
                y_true_temp = np.array(y_true_temp).reshape(-1, 1)
                input_N[0] = np.dot(self.weights[0], x) + self.biases[0]
                sigmoid_N[0] = sigmoid(input_N[0])
                for i in range(1, len_layers):
                    input_N[i] = np.dot(self.weights[i], sigmoid_N[i - 1]) + self.biases[i]
                    sigmoid_N[i] = sigmoid(input_N[i])

                
                d_L_d_ypred = -2 * (y_true_temp - sigmoid_N[len_layers - 1])

                d_ypred_d_w[len_layers - 1] = np.dot(d_L_d_ypred * deriv_sigmoid(input_N[len_layers - 1]), sigmoid_N[len_layers - 1].T)
                d_ypred_d_b[len_layers - 1] = d_L_d_ypred * deriv_sigmoid(input_N[len_layers - 1])
                d_temp = np.dot(self.weights[len_layers - 1].T, d_L_d_ypred * deriv_sigmoid(sigmoid_N[len_layers - 1]))
                if len_layers > 1:
                    for i in range(len_layers - 2, 0, -1):
                        d_ypred_d_w[i] = np.dot(d_temp * deriv_sigmoid(input_N[i]), sigmoid_N[i - 1].T)
                        d_ypred_d_b[i] = d_temp * deriv_sigmoid(input_N[i])
                        d_temp = np.dot(self.weights[i].T, d_temp * deriv_sigmoid(sigmoid_N[i]))
                d_ypred_d_w[0] = np.dot(d_temp * deriv_sigmoid(input_N[0]), x.T)
                d_ypred_d_b[0] = d_temp * deriv_sigmoid(input_N[0])

                for i in range(len_layers-1):
                    self.weights[i] -= learning_rate * d_ypred_d_w[i]
                    self.biases[i] -= learning_rate * d_ypred_d_b[i]

            if epoch % 1 == 0:
                print(all_y_trues.shape)
                all_y_trues_temp = np.eye(10)[np.array(all_y_trues, dtype=int)]
                y_preds = np.array([self.feedforward(x) for x in data])
                print(all_y_trues_temp.shape, y_preds.shape)
                loss = mse_loss(all_y_trues_temp, y_preds)
                print(f"Epoch {epoch} loss: {loss:.4f}")
        save_dict = {}
        for idx, W in enumerate(self.weights):
            save_dict[f'w{idx}'] = W
        for idx, b in enumerate(self.biases):
            save_dict[f'b{idx}'] = b
        np.savez('neural_network_weights.npz', **save_dict)

    def feedforward(self, x):
        len_layers = len(self.layers_size) - 1
        input_N = [np.array([]) for _ in range(len_layers)]
        sigmoid_N = [np.array([]) for _ in range(len_layers)]
        if os.path.exists('neural_network_weights.npz'):
            data = np.load('neural_network_weights.npz')
            self.weights = [data[f'w{i}'] for i in range(len(self.layers_size) - 1)]
            self.biases  = [data[f'b{i}'] for i in range(len(self.layers_size) - 1)]
        x = np.array(x).reshape(-1, 1)
        input_N[0] = np.dot(self.weights[0], x) + self.biases[0]
        sigmoid_N[0] = sigmoid(input_N[0])
        for i in range(1, len_layers):
            input_N[i] = np.dot(self.weights[i], sigmoid_N[i - 1]) + self.biases[i]
            sigmoid_N[i] = sigmoid(input_N[i])
        return sigmoid_N[len_layers - 1].flatten()

if __name__ == "__main__":
    dataset = openml.datasets.get_dataset(554)
    X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe")

    df = X.copy()
    if y is not None:
        df[dataset.default_target_attribute] = y

    data = df.loc[0:59999, df.columns != 'class'].copy()
    data = data.to_numpy().astype(float)
    data = data / 255.0
    all_y_trues = df.loc[0:59999, 'class'].copy()
    all_y_trues = all_y_trues.to_numpy().astype(float)
    network = NeuralNetwork()
    network.train(data, all_y_trues)

    def predict(network, number):
        predictions = network.feedforward(number)
        for i, pred in enumerate(predictions):
            print(f"Вероятность того, что это число {i} = {pred:.3f}")
        print('\n')

    F_data = df.loc[405:410, df.columns != 'class'].copy()
    F_data = F_data.to_numpy().astype(float)
    F_data = F_data / 255.0

    for number in F_data:
        predict(network, number)
    print(df.loc[405:410, 'class'].to_numpy().astype(float))

