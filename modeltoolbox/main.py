from typing import Union, Tuple, Any

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import os
import threading
from contextlib import contextmanager
import multiprocessing

_local = threading.local()
counts = 1


def preview(func):
    def inner(*args, **kwargs):
        func(*args, **kwargs)
        plt.show()

    return inner


def save_plt_fig(path="./pypic", hold="off"):
    """保存plt中的图片，按数字顺序为图片命名，默认保存到E:\图片\pypic，可手动指定保存路径（绝对路径！！）
    另：程序默认画完一张图自动hold off，如需设置hold on需要给hold传入参数"on"

    Args:
        path:
        hold:
    """
    global counts
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(f"{path}/{counts}.jpg")
    counts += 1
    if hold == "on":
        pass
    else:
        plt.close()


@preview
def SavePreview_plt_fig(*args, **kwargs):
    """SavePreview_plt_fig.
    本函数所有参数都和save_plt_fig相同，详情参见前者
    在save_plt_fig的基础上利用装饰器实现在保存后展示图片，plt的默认设置好像是执行plt.show()
    以后自动hold=off，所以这里处理关键字hold没什么意义，在plt那里最后的结果都跟hold=off一样的，
    但是写都写了，再说万一以后有别的解决办法呢？就不删了

    Args:
        args:
        kwargs:
    """
    track = 1
    if "hold" in kwargs.keys():
        if kwargs["hold"] == "on":
            track = 0
        kwargs["hold"] = "on"
    save_plt_fig(*args, **kwargs)
    if track == 1:
        plt.close()


@contextmanager
def acquire(*locks):
    """acquire.
    这个是抄的网上的锁管理器

    Args:
        locks:
    """
    locks = sorted(locks, key=lambda x: id(x))

    acquired = getattr(_local, 'acquired', [])
    if acquired and max(id(lock) for lock in acquired) >= id(locks[0]):
        raise RuntimeError('Lock Order Violation')

    acquired.extend(locks)
    _local.acquired = acquired

    try:
        for lock in locks:
            lock.acquire()
        yield
    finally:
        for lock in reversed(locks):
            lock.release()
        del acquired[-len(locks):]


cpu_nums = os.cpu_count()


def grid_caculator_multiprocessing(x,
                                   y,
                                   calculator,
                                   title='pic',
                                   xlabel='x',
                                   ylabel='y',
                                   zlabel='z',
                                   useMatlab=True,
                                   n_jobs=None):
    """mesh_multiprocessing.本函数与下方的calculate函数共同组成并行计算函数体
       用法：mp.mesh_multiprocessing(eng, x, y ,' x * np.exp(-x * 2 - y**2)')

    Args:
        x:
        y:
        calculator:
        title:
        xlabel:
        ylabel:
        zlabel:
        useMatlab:
        n_jobs:并行数，默认为cpu核心数量
    """
    global mpmesh_lsts, cpu_nums, mpmesh_x, mpmesh_y, mpmesh_caculator
    if n_jobs is None:
        n_jobs = cpu_nums
    mpmesh_lsts = np.array_split(x, n_jobs)
    if isinstance(calculator, str):
        mpmesh_caculator = eval(f"lambda x,y:{calculator}")
    else:
        mpmesh_caculator = calculator
    mpmesh_y = y
    mpmesh_x = x
    with multiprocessing.Pool(n_jobs) as p:
        stack = np.hstack(p.map(calculate, mpmesh_lsts))
    # if useMatlab:
    #     mpmesh_stacked = stack
    #     mesh(eng, x, y, mpmesh_stacked)
    #     eng.ylabel(ylabel)
    #     eng.grid("on", nargout=0)
    #     eng.title(title)
    #     eng.xlabel(xlabel)
    #     eng.zlabel(zlabel)
    # else:
    # return stack
    return stack


def calculate(x):
    global mpmesh_y
    x, y = np.meshgrid(x, mpmesh_y)
    return mpmesh_caculator(x, y)


def change_col_dtype(DataFrame, before, after):
    """boolcol_to_int.将给定的DateFrame中的所有某(before)类型的列转化为其他(after)类型

    Args:
        DataFrame:
        before: 转化前的类型名
        after: 转化后的类型名
    """
    for column in DataFrame.columns:
        if DataFrame[column].dtype == before:
            DataFrame[column] = DataFrame[column].astype(after)
    return DataFrame


def corr_heatmap(DataFrame, title='pic'):
    """heatmap.快速绘制出一个含有数字的DataFrame的相关系数热力图

    Args:
        DataFrame: pd.DataFrame
        title:
    """
    from seaborn import heatmap
    DataFrame = change_col_dtype(DataFrame, bool, int)
    numeric_columns = DataFrame.select_dtypes(include=['number'])
    heatmap(numeric_columns.corr(), annot=True)
    plt.title(title)
    plt.show()


def sklearn_model_report(model, train_data, label, scoring='accuracy'):
    """sklearn_model_report.本函数用于输出已训练好的sklearn模型的各项性能参数

    Args:
        model: 已训练好的模型
        train_data: 数据集（不含结果）
        label: train_data对应的正确结果
        scoring: 最后要输出的参数的名称，如accuracy, precision, recall，如果只想求一个就输入字符串，否则用一个列表框起来
    """
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.model_selection import cross_val_score
    pred = model.predict(train_data)
    print('混淆矩阵如下：')
    print(confusion_matrix(label, pred), '\n')
    print('查全率查准率等各项指标如下：')
    print(classification_report(label, pred))

    if isinstance(scoring, list):
        for i in scoring:
            _train = cross_val_score(model, train_data, label, scoring=i)
            print(f"在给出的训练集上，本模型{i}指标的多次预测平均值为：{_train.mean()}")

    if isinstance(scoring, str):
        _train = cross_val_score(model, train_data, label, scoring=scoring)
        print(f"在给出的训练集上，本模型{scoring}指标的多次预测平均值为：{_train.mean()}")


def general_clf_report(predicted_data, label):
    from sklearn.metrics import confusion_matrix, classification_report
    if not isinstance(predicted_data, list):
        print("混淆矩阵如下")
        print(confusion_matrix(label, predicted_data), '\n')
        print('查全率查准率等各项指标如下：')
        print(classification_report(label, predicted_data))
    else:
        ...
