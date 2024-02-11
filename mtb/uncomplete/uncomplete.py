import numpy as np
import pandas as pd


class Grey_model_21:
    def __init__(self):
        ...

    def fit(self, x):
        if isinstance(x, list):
            self.x = np.array(x)
        elif isinstance(x, np.ndarray):
            self.x = x
        elif isinstance(x, pd.Series):
            self.x = x.values

    def predict(self, n):
        x1, a1x0 = [], []
        tmp = 0
        for i in self.x:
            tmp += i
            x1.append(tmp)
        x1 = np.array(x1)

        a1x0 = self.x[1:] - self.x[:-1]
        Z = -0.5 * (x1[1:] + x1[:-1])

        B = np.c_[-self.x[1:].T, Z.T, np.ones(len(Z)).T]
        Y = a1x0.T
        u = np.linalg.inv((B.T.dot(B))).dot(B.T).dot(Y)
        a1 = u[0]
        a2 = u[1]
        b = u[2]
        print(a1, a2, b)

    def get_report(self):
        ...
