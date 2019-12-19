from matplotlib import pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

from joblib import dump

filename = 'example_model.joblib'

digits = datasets.load_digits()

images_labels = list(zip(digits.images, digits.target))

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

classifier = svm.SVC(gamma=0.001)

x_train, x_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)

classifier.fit(x_train, y_train)

dump(classifier, filename)

print('Model saved as %s' % filename)
