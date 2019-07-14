import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes):
        weight_shapes = [(a, b)
                         for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]
        self.weights = [np.random.standard_normal(
            s)/s[1]**.5 for s in weight_shapes]
        self.biases = [np.zeros((s, 1)) for s in layer_sizes[1:]]

    def predict(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.activation(np.matmul(w, a) + b, 'sigmoid')
        return a

    def get_accuracy(self, images, labels):
        predictions = self.predict(images)
        num_correct = sum([np.argmax(prediction) == np.argmax(label)
                           for prediction, label in zip(predictions, labels)])
        return num_correct, len(images), (num_correct/len(images)*100)

    @staticmethod
    def activation(x, act_func):
        if act_func == 'relu':
            return np.maximum(x, np.zeros(x.shape))
        elif act_func == 'sigmoid':
            return 1/(1+np.exp(-x))
