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
            a = self.activation(np.matmul(w, a) + b)
        return a

    def train(self, images, labels):
        predictions = self.predict(images)
        labeled_predictions = [(np.argmax(prediction), np.argmax(label))
                               for prediction, label in zip(predictions, labels)]
        error = [(desired - guess) * .01 for guess,
                 desired in labeled_predictions]
        new_weights = [error + w for w in self.weights]
        print(error)

    def get_accuracy(self, images, labels):
        predictions = self.predict(images)
        num_correct = sum([np.argmax(prediction) == np.argmax(label)
                           for prediction, label in zip(predictions, labels)])
        return num_correct, len(images), (num_correct/len(images)*100)

    @staticmethod
    def activation(x):
        return 1/(1+np.exp(-x))
