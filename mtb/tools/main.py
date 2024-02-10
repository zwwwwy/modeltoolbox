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
    # ubuntu下的系统自带字体
    plt.rcParams['font.sans-serif'] = ['AR PL UMing CN']
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    pd.set_option('display.max_columns', None)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)


def jiecheng(x):
    return 1 if x <= 1 else jiecheng(x - 1) * x


def C(a, b):
    sum = 1
    for i in range(b):
        sum *= a
        a -= 1
    return sum / jiecheng(b)


def A(a, b):
    sum = 1
    for i in range(b):
        sum *= a
        a -= 1
    return sum

__all__ = ['timersss']
