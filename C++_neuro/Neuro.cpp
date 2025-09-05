#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <array>
#include <sstream>
#include <string>
#include <omp.h>  

class NeuralNetwork {
private:
    const std::array<int, 4> layers_size = {784, 16, 16, 10};
    std::vector<std::vector<float>> neurons;
    std::vector<std::vector<std::vector<float>>> weights;
    std::vector<std::vector<float>> biases;
    std::vector<std::vector<float>> errors; 
    std::vector<std::vector<float>> sigmoid_derivatives; 
    const std::string modelFile = "neural_model.bin";  

    void saveModel() {
        std::ofstream out(modelFile, std::ios::binary);
        if (!out) {
            std::cerr << "Error opening file for saving model!" << std::endl;
            return;
        }
        size_t numLayers = weights.size();
        out.write(reinterpret_cast<const char*>(&numLayers), sizeof(numLayers));
        for (const auto& layer : weights) {
            size_t layerSize = layer.size();
            out.write(reinterpret_cast<const char*>(&layerSize), sizeof(layerSize));
            for (const auto& neuron : layer) {
                size_t neuronSize = neuron.size();
                out.write(reinterpret_cast<const char*>(&neuronSize), sizeof(neuronSize));
                out.write(reinterpret_cast<const char*>(neuron.data()), neuron.size() * sizeof(float));
            }
        }
        for (const auto& bias : biases) {
            size_t biasSize = bias.size();
            out.write(reinterpret_cast<const char*>(&biasSize), sizeof(biasSize));
            out.write(reinterpret_cast<const char*>(bias.data()), bias.size() * sizeof(float));
        }
        out.close();
        std::cout << "Model saved to " << modelFile << std::endl;
    }

    void loadModel() {
        std::ifstream in(modelFile, std::ios::binary);
        if (!in) {
            std::cout << "No saved model found, using random initialization." << std::endl;
            return;
        }
        size_t numLayers;
        in.read(reinterpret_cast<char*>(&numLayers), sizeof(numLayers));
        if (numLayers != weights.size()) {
            std::cerr << "Model file mismatch, using random initialization." << std::endl;
            return;
        }
        for (auto& layer : weights) {
            size_t layerSize;
            in.read(reinterpret_cast<char*>(&layerSize), sizeof(layerSize));
            if (layerSize != layer.size()) {
                std::cerr << "Model file mismatch, using random initialization." << std::endl;
                return;
            }
            for (auto& neuron : layer) {
                size_t neuronSize;
                in.read(reinterpret_cast<char*>(&neuronSize), sizeof(neuronSize));
                if (neuronSize != neuron.size()) {
                    std::cerr << "Model file mismatch, using random initialization." << std::endl;
                    return;
                }
                in.read(reinterpret_cast<char*>(neuron.data()), neuron.size() * sizeof(float));
            }
        }
        for (auto& bias : biases) {
            size_t biasSize;
            in.read(reinterpret_cast<char*>(&biasSize), sizeof(biasSize));
            if (biasSize != bias.size()) {
                std::cerr << "Model file mismatch, using random initialization." << std::endl;
                return;
            }
            in.read(reinterpret_cast<char*>(bias.data()), bias.size() * sizeof(float));
        }
        in.close();
        std::cout << "Model loaded from " << modelFile << std::endl;
    }

public:
    NeuralNetwork() {
        neurons.resize(layers_size.size());
        for (size_t i = 0; i < layers_size.size(); ++i) {
            neurons[i].resize(layers_size[i]);
        }
        weights.resize(layers_size.size() - 1);
        for (size_t i = 0; i < layers_size.size() - 1; ++i) {
            weights[i].resize(layers_size[i]);
            for (size_t j = 0; j < layers_size[i]; ++j) {
                weights[i][j].resize(layers_size[i + 1]);
                for (size_t k = 0; k < layers_size[i + 1]; ++k) {
                    weights[i][j][k] = (rand() % 2000 - 1000) / 1000.0f;
                }
            }
        }
        for(size_t i = 0; i < layers_size.size() - 1; ++i) {
            biases.push_back(std::vector<float>(layers_size[i + 1], 0.0f));
        }
        for(int i = 0; i < weights.size(); i++) {
            std::cout << i << " layer:" << weights[i].size() << ":" << weights[i][0].size() << std::endl;
        }
        errors.resize(layers_size.size());
        for (size_t i = 0; i < layers_size.size(); ++i) {
            errors[i].resize(layers_size[i], 0.0f);
        }
        loadModel();  
    }
    float sigmoid(float x) {
        return 1.0f / (1.0f + expf(-x));
    }
    float sigmoidDerivative(float x) {
        float s = sigmoid(x);
        return s * (1.0f - s);
    }
    float mse_loss(const std::vector<float>& predicted, const std::vector<float>& actual) {
        float sum = 0.0f;
        for (size_t i = 0; i < predicted.size(); ++i) {
            float diff = predicted[i] - actual[i];
            sum += diff * diff;
        }
        return sum / predicted.size();
    }
    std::vector<float> feedforward(const std::vector<float>& input) {
        sigmoid_derivatives.resize(layers_size.size());
        for (size_t i = 0; i < layers_size.size(); ++i) {
            sigmoid_derivatives[i].resize(layers_size[i], 0.0f);
        }
        for(size_t i = 0; i < layers_size.size() - 1; ++i){
            #pragma omp parallel for
            for(size_t j = 0; j < layers_size[i+1]; ++j){
                for(size_t k = 0; k < layers_size[i]; ++k){
                    neurons[i+1][j] += neurons[i][k] * weights[i][k][j];
                }
                neurons[i+1][j] += biases[i][j];
                neurons[i+1][j] = sigmoid(neurons[i+1][j]);
                sigmoid_derivatives[i+1][j] = sigmoidDerivative(neurons[i+1][j]);
            }
        }
        return neurons.back();
    }
    void train(const std::vector<std::vector<float>>& inputs, const std::vector<float>& targets, int epochs = 3, float learning_rate = 0.001f, int batch_size = 32) {
        if (inputs.empty() || targets.empty() || inputs.size() != targets.size()) {
            std::cerr << "Invalid training data!" << std::endl;
            return;
        }
        std::vector<std::vector<float>> one_hot_targets(targets.size(), std::vector<float>(10, 0.0f));
        for (size_t i = 0; i < targets.size(); ++i) {
            int digit = static_cast<int>(targets[i]);
            if (digit >= 0 && digit < 10) {
                one_hot_targets[i][digit] = 1.0f;
            }
        }
        std::cout << "Beginning training for " << epochs << " epochs with batch size " << batch_size << "..." << std::endl;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;
            size_t num_batches = (inputs.size() + batch_size - 1) / batch_size;
            for (size_t batch = 0; batch < num_batches; ++batch) {
                std::vector<std::vector<std::vector<float>>> grad_weights(weights.size());
                for (size_t i = 0; i < weights.size(); ++i) {
                    grad_weights[i].resize(weights[i].size());
                    for (size_t j = 0; j < weights[i].size(); ++j) {
                        grad_weights[i][j].assign(weights[i][j].size(), 0.0f);
                    }
                }
                std::vector<std::vector<float>> grad_biases(biases.size());
                for (size_t i = 0; i < biases.size(); ++i) {
                    grad_biases[i].assign(biases[i].size(), 0.0f);
                }
                float batch_loss = 0.0f;
                size_t start_idx = batch * batch_size;
                size_t end_idx = std::min(start_idx + batch_size, inputs.size());
                #pragma omp parallel for reduction(+:batch_loss)
                for (size_t example = start_idx; example < end_idx; ++example) {
                    for (auto& layer : neurons) {
                        std::fill(layer.begin(), layer.end(), 0.0f);
                    }
                    for (size_t i = 0; i < inputs[example].size() && i < neurons[0].size(); ++i) {
                        neurons[0][i] = inputs[example][i];
                    }
                    feedforward(inputs[example]);
                    batch_loss += mse_loss(neurons.back(), one_hot_targets[example]);
                    for (auto& layer : errors) {
                        std::fill(layer.begin(), layer.end(), 0.0f);
                    }
                    for (size_t j = 0; j < layers_size.back(); ++j) {
                        errors.back()[j] = neurons.back()[j] - one_hot_targets[example][j];
                    }
                    for (int i = layers_size.size() - 2; i >= 0; --i) {
                        for (int j = 0; j < layers_size[i]; ++j) {
                            float error_sum = 0.0f;
                            for (int k = 0; k < layers_size[i+1]; ++k) {
                                error_sum += weights[i][j][k] * errors[i+1][k] * sigmoid_derivatives[i+1][k];
                            }
                            errors[i][j] = error_sum;
                        }
                    }
                    #pragma omp critical
                    {
                        for (int i = layers_size.size() - 2; i >= 0; --i) {
                            for (int j = 0; j < layers_size[i+1]; ++j) {
                                float delta = errors[i+1][j] * sigmoid_derivatives[i+1][j];
                                for (int k = 0; k < layers_size[i]; ++k) {
                                    grad_weights[i][k][j] += neurons[i][k] * delta;
                                }
                                grad_biases[i][j] += delta;
                            }
                        }
                    }
                }
                for (size_t i = 0; i < weights.size(); ++i) {
                    for (size_t j = 0; j < weights[i].size(); ++j) {
                        for (size_t k = 0; k < weights[i][j].size(); ++k) {
                            weights[i][j][k] -= learning_rate * grad_weights[i][j][k] / (end_idx - start_idx);
                        }
                    }
                }
                for (size_t i = 0; i < biases.size(); ++i) {
                    for (size_t j = 0; j < biases[i].size(); ++j) {
                        biases[i][j] -= learning_rate * grad_biases[i][j] / (end_idx - start_idx);
                    }
                }
                total_loss += batch_loss;
                if ((batch + 1) % 100 == 0) {
                    std::cout << "Epoch " << epoch + 1 << ", batch " << batch + 1 << "/" << num_batches << "\r";
                    std::cout.flush();
                }
            }
            float avg_loss = total_loss / inputs.size();
            std::cout << "Epoch " << epoch + 1 << " completed, average loss: " 
                      << avg_loss << "                  " << std::endl;
        }
        saveModel();
        std::cout << "Training completed." << std::endl;
    }

};


int main() {
    omp_set_num_threads(4);  // Установлено для 4 потоков
    std::ifstream arffFile("mnist_784.arff");
    if (!arffFile.is_open()) {
        return 0;
    }
    float cell_val;
    std::string line;
    std::vector<std::vector<float>> data;
    std::vector<float> data_target;
    while (std::getline(arffFile, line)) {
        if (line.empty()) {
            continue;
        }
        if (line.rfind("@DATA", 0) == 0) {
            continue;
        }
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;
        while (std::getline(ss, cell, ',')) {
            cell_val = std::stof(cell);
            row.push_back(cell_val/255.0f);
        }
        if (!row.empty()) {
            data_target.push_back(row.back());
            row.pop_back();  
        }
        data.push_back(row);
    }
    arffFile.close();

    NeuralNetwork nn;
    std::cout << "training?  y-yes/enter-no" << std::endl;
    if (std::cin.get() == 'y') {
        nn.train(data, data_target);
    }
    std::vector<float> neurons = nn.feedforward(data[0]);
    for (size_t i = 0; i < neurons.size(); ++i) {
        std::cout << "Number " << i << ": " << neurons[i] << std::endl;
    }
    
    std::cin.get();
    return 0;
}