import numpy as np

_NEPOCH = 20


class softmax_descent():
    def __init__(self, X, M, W, eta, batchSize=None, chatol=None, testSet=True, beta=None):

        self.X = X
        self.eta = eta
        self.gamma = 0.000
        self.TOT = M[:, -1].sum()
        self.M = M / self.TOT
        self.W = W
        if beta is None:
            self.beta = np.zeros((M.shape[0], X.shape[1]))
            self.beta[M == 0] = - 10 ** (10)
        else:
            self.beta = beta
        self.H = self.beta.shape[0]
        self.testSet = testSet
        self.theta2 = 0
        self.theta1 = np.zeros(self.beta.shape)

        if batchSize == None:
            self.batchSize = 1
        else:
            self.batchSize = batchSize

        if chatol == None:
            self.chatol = 0.01
        else:
            self.chatol = chatol

    def expand_sample(self, X, W):
        return np.repeat(X, W, axis=0)

    def next_batch(self, batchSize, X):

        for i in np.arange(0, X.shape[0], batchSize):
            # adjust size at the end of the sample
            end_batch = min([i + batchSize, X.shape[0]])

            # yield a tuple of the current batched data and labels
            yield X[i:end_batch]

    def reshuffle(self):
        X = self.expand_sample(self.X, self.W)
        np.random.shuffle(X)
        return X

    def computeProb(self, X, beta):

        XE = np.exp(np.dot(X, beta))
        XESUM = np.sum(XE, axis=1)

        return XE / XESUM[:, np.newaxis]

    def gradient(self, X, beta, M, W=None):

        SX = np.transpose(self.computeProb(X, beta))
        if W is not None:
            X = np.multiply(X, W[:, np.newaxis])
        return np.dot(SX, X) / X[:, -1].sum() - M + 10 ** (-12) * np.transpose(beta)

    def updatebeta(self, X):

        beta = self.beta - 0.8 * self.gamma
        g = self.gradient(X, np.transpose(beta), self.M)

        gamma = 0.8 * self.gamma + self.eta * g
        beta = beta - gamma

        self.gamma = gamma
        self.beta = beta

    def iterateBatch(self):
        X = self.reshuffle()
        i = 0
        for XB in self.next_batch(self.batchSize, X):
            # if i % 500 == 0:
            # print(self.computeLoss())
            self.updatebeta(XB)
            i = i + 1

    def epochIter(self):

        GRADIENT_TRAINING = np.zeros(_NEPOCH)
        LOSS = np.zeros(_NEPOCH)

        for i in np.arange(1, _NEPOCH + 1):
            LOSS[i - 1] = self.computeLoss()
            GRADIENT_TRAINING[i - 1] = np.sum(
                self.gradient(self.X, np.transpose(self.beta), self.M, W=self.W)[:, 0] ** 2)

            if (i >= 2) & (LOSS[i - 1] / LOSS[i - 2] >= 1 - self.chatol) | (LOSS[i - 1] <= 1.6):
                break  # stop conditions

            self.iterateBatch()
            self.eta = self.eta / (i + 1) * i

        return GRADIENT_TRAINING, LOSS

    def computeLoss(self):

        # estimated prob
        SX = self.computeProb(self.X, np.transpose(self.beta))
        X = np.multiply(self.X, self.W[:, np.newaxis])
        MQ = np.dot(np.transpose(SX), X) / X[:, -1].sum()
        NQ = MQ[:, -1]

        NP = self.M[:, -1]
        MQ = np.divide(MQ, NQ[:, np.newaxis])
        MP = np.divide(self.M, NP[:, np.newaxis])

        return np.sqrt(np.mean((MQ - MP) ** 2)) * 100

    def predict(self, beta):

        # estimated prob
        SX = self.computeProb(self.X, np.transpose(beta))
        SX = np.multiply(SX, self.W[:, np.newaxis])

        # estimated number of households
        NX = np.sum(SX, axis=0)

        # proba to draw each household
        SX = np.divide(SX, NX[np.newaxis, :])

        # draw households for each block group
        EM = np.zeros(self.M.shape)
        for i in np.arange(self.M.shape[0]):
            P = SX[:, i]
            H = np.random.choice(np.arange(0, self.X.shape[0]), int(self.M[i, -1] * self.TOT), p=P, replace=True)

            # computed simulated marginal
            XH = X[H, :]
            EM[i, :] = np.sum(XH, axis=0)

        del SX
        return np.sqrt(np.mean((EM - self.M * self.TOT) ** 2))


if __name__ == '__main__':
    import os
    import time
    import matplotlib.pyplot as plt
    import pandas as pd

    input_directory = 'C:\\Users\\xgitiaux\\Documents\\Research and Analysis\\Synthetizer\\Data\\Inputs'
    filename_x = os.path.join(input_directory, 'sample.npy')
    filename_m = os.path.join(input_directory, 'marginal.npy')
    filename_w = os.path.join(input_directory, 'weight.npy')

    output_directory = 'C:\\Users\\xgitiaux\\Documents\\Research and Analysis\\Synthetizer\\Data\\Outputs'
    filename_beta = os.path.join(output_directory, 'beta.npy')
    beta = np.load(filename_beta)

    X = np.load(filename_x)
    W = np.load(filename_w)
    M = np.load(filename_m)
    MH = M[:, -1]
    M = M[MH > 0, :]
    # M[M == 0] = 0.0000000000001

    print("starting SGD")

    sf = softmax_descent(X, M, W, 1.0, batchSize=256)
    s = time.time()
    print(sf.predict(beta))
    gradient_m, loss = sf.epochIter()
    print("time elapsed: {:.2f}s".format(time.time() - s))

    output_directory = 'C:\\Users\\xgitiaux\\Documents\\Research and Analysis\\Synthetizer\\Data\\Outputs'
    filename_beta = os.path.join(output_directory, 'beta.npy')
    beta = sf.beta
    np.save(filename_beta, beta)

    filename_loss = os.path.join(output_directory, 'loss.npy')
    np.save(filename_loss, loss)

    plt.plot(np.arange(_NEPOCH), gradient_m, 'r-')
    plt.show()

    plt.plot(np.arange(_NEPOCH), loss, 'b-')
    plt.show()
