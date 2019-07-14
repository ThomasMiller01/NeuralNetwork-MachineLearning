import neuralnetwork as nn
import numpy as np

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']


layer_sizes = (training_images.shape[1], 5, 10)

net = nn.NeuralNetwork(layer_sizes)

num_correct, num_total, num_accuracy = net.get_accuracy(
    training_images, training_labels)

print('{0}/{1} accuracy: {2}%'.format(num_correct, num_total, num_accuracy))
