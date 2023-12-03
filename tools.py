from time import time
import prettytable as pt
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO


def timer(MeasuredFunction):
    def deco(*args):
        t1 = time()
        MeasuredFunction(*args)
        t2 = time()
        print(f"{MeasuredFunction}的运行时间为{t2-t1}s")

    return deco


def printable(dataFrame):
    output = StringIO()
    dataFrame.to_csv(output)
    output.seek(0)
    result = pt.from_csv(output)
    print(result)


def prefer_settings():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    pd.set_option('display.max_columns', None)
