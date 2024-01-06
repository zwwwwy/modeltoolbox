import numpy as np


class GridFunc:
    def __init__(self):
        self.x = np.arange(-2, 2.1, 0.01)
        self.y = np.arange(-3, 3.1, 0.01)
        self.meshed_x, self.meshed_y = np.meshgrid(self.x, self.y)
        self.z = self.meshed_x * np.exp(-(self.meshed_x**2) - self.meshed_y**2)
        self.expression = "x*np.exp(-x**2-y**2)"

    def print(self):
        print("函数表达式为z=x*exp(-x**2-y**2)\nx的范围是(-2,2)\ny的范围是(-3,3)")


class LineFunc:
    def __init__(self):
        self.x = np.linspace(-np.pi, np.pi, 100)
        self.y = np.sin(self.x)

    def print(self):
        print("函数表达式为y=sin(x)\nx的范围是(-pi,pi)")


class GridOperation:
    def __init__(self):
        self.x = np.linspace(0.01,0.99,100)
        self.y = self.x.copy()

    def tmp(self,C, a1, q):
        C0 = C
        k = 0
        a = a1
        while True:
            k += 1
            tmp = C * a
            C = C - tmp
            a = a * q
            if C / C0 <= 0.001:
                return k
            if k >= 1000:
                return -1

    def operator(self, x=None, y=None):
        if x is None or y is None:
            x=self.x
            y=self.y
        z = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(x[0])):
                z[i][j] = self.tmp(21, x[i][j], y[i][j])
        return z


