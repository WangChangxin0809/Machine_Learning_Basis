import numpy as np
import matplotlib.pyplot as plt

class LinearRegression(object):

    def __init__(self,learning_rate = 0.01, max_iter = 100, seed = None):
        np.random.seed(seed)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.w = np.random.normal(0,1)
        self.b = np.random.normal(0,1)
        self.loss_list = []

    def __f(self, x, w, b):
        return w * x + b

    def predict(self, x):
        y_pred = self.__f(x, self.w, self.b)
        return y_pred

    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def fit(self, x, y):
        for i in range(self.max_iter):
            y_pred = self.predict(x)
            self.loss_list.append(self.loss(y, y_pred))
            self.__train_step(x, y)

    def __calc_gradient(self, x, y):
        d_w = np.mean(2 * (x * self.w + self.b - y) * x)
        d_b = np.mean(2 * (x * self.w + self.b - y))
        return d_w,d_b

    def __train_step(self, x, y):
        d_w, d_b = self.__calc_gradient(x, y)
        self.w = self.w - self.learning_rate * d_w
        self.b = self.b - self.learning_rate * d_b
        return self.w, self.b

np.random.seed(34223425)
data_size = 100
x = np.random.uniform(low = 1.0, high = 10.0, size = data_size)
y = x * 20 + 10 + np.random.normal(loc = 0.0, scale = 10.0, size = data_size)

shuffled_index = np.random.permutation(data_size)
x = x[shuffled_index]
y = y[shuffled_index]

split_index = int(data_size * 0.7)
x_train = x[:split_index]
x_test = x[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

regr = LinearRegression(learning_rate = 0.01, max_iter = 10, seed = 0)
regr.fit(x_train, y_train)
print('w: \t{:.3}'.format(regr.w))
print('b: \t{:.3}'.format(regr.b))

def show_data(x, y, w = None, b = None):
    plt.scatter(x, y, marker = '.')
    if w is not None and b is not None:
        plt.plot(x, w * x + b, c = 'red')
    plt.show()

show_data(x, y, regr.w, regr.b)

plt.plot(np.arange(len(regr.loss_list)), regr.loss_list, marker = 'o',c = 'green')

plt.show()

y_test_pred = regr.predict(x_test)
mse_loss = regr.loss(y_test,y_test_pred)
print('Loss on test set : \t{:.3}'.format(mse_loss))

import numpy as np
import matplotlib.pyplot as plt


