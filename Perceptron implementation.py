import numpy as np

class Perceptron(object):
    """ Perceptron classifier

    Parameters
    eta: float - learning rate (between 0.0 and 1.0)
    n_iter: int - passes over the training dataset
    random_state: int - random number generator seed for random weight initialization.

    Attributes
    w_: 1d-array - weights after fitting
    errors_: list - number of misclassifications (updates) in each epoch
    """

    def __init__(self, eta=0.01, n_inter=50, random_state=1):
        self.eta = eta
        self.epochs = n_inter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit training data

        :param X: {array-like}, shape = {n_examples, n_features}
        Training vectors, where n_examples is the number of examples and n_features is
        the number of examples and n_features is the number of features
        :param y: array-like, shape = {n_examples}
        Training values.
        :return: self:object
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Caculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)