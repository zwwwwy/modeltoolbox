import os
import threading
from contextlib import contextmanager
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

_local = threading.local()
counts = 1


def preview(func):
    def inner(*args, **kwargs):
        func(*args, **kwargs)
        plt.show()

    return inner


def save_plt_fig(path="./pypic", hold="off"):
    """保存plt中的图片，按数字顺序为图片命名，默认保存到E:\\图片\\pypic，可手动指定保存路径（绝对路径！！）
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

    acquired = getattr(_local, "acquired", [])
    if acquired and max(id(lock) for lock in acquired) >= id(locks[0]):
        raise RuntimeError("Lock Order Violation")

    acquired.extend(locks)
    _local.acquired = acquired

    try:
        for lock in locks:
            lock.acquire()
        yield
    finally:
        for lock in reversed(locks):
            lock.release()
        lenth = len(locks)
        del acquired[lenth:]


cpu_nums = os.cpu_count()


def grid_caculator_multiprocessing(
    x,
    y,
    calculator,
    title="pic",
    xlabel="x",
    ylabel="y",
    zlabel="z",
    draw_pic=True,
    n_jobs=None,
):
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
    if draw_pic:
        fig = go.Figure(data=[go.Surface(x=x, y=y, z=stack)])

        fig.update_layout(
            title=title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel  # 标题
        )

        fig.show()
    else:
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


def corr_heatmap(DataFrame, title="pic"):
    """heatmap.快速绘制出一个含有数字的DataFrame的相关系数热力图

    Args:
        DataFrame: pd.DataFrame
        title:
    """
    from seaborn import heatmap

    DataFrame = change_col_dtype(DataFrame, bool, int)
    numeric_columns = DataFrame.select_dtypes(include=["number"])
    heatmap(numeric_columns.corr(), annot=True)
    plt.title(title)
    plt.show()


def fast_corrscatter_evaluate(DataFrame, target, n=4):
    """fast_corrscatter_evaluate.快速绘制相关系数散点图（挑和target关联最大的四个）

    Args:
        DataFrame:
        target:
    """
    from pandas.plotting import scatter_matrix

    DataFrame = change_col_dtype(DataFrame, bool, int)
    numeric_columns = DataFrame.select_dtypes(include=["number"])
    attribute = numeric_columns.corr()[target].nlargest(n).index.tolist()
    scatter_matrix(DataFrame[attribute])


def sklearn_model_report(model, train_data, label, scoring="accuracy"):
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
    print("混淆矩阵如下：")
    print(confusion_matrix(label, pred), "\n")
    print("查全率查准率等各项指标如下：")
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
        print(confusion_matrix(label, predicted_data), "\n")
        print("查全率查准率等各项指标如下：")
        print(classification_report(label, predicted_data))
    else:
        ...


def confusion_matrix_analysis(confusion_matrix):
    """confusion_matrix_analysis. 输出俩图，第一个图是混淆矩阵的热力图
    第二个图里面每一行是代表准确值，每一列代表预测值，所以每一个格子里的值代表某一准确值被预测为某错误的预测值的概率
    返回precision, recall, false_positive_rate

    Args:
        confusion_matrix:
    Returns:
        precision: 查准率/精度
        recall: 查全率/召回率/真正例率
        false_positive_rate: 假正例率
        f1_score: f1分数
        accuracy: 准确率
        macro_precision: 宏查准率
        macro_recall: 宏查全率
        macro_f1: 宏F1
        micro_precision: 微查准率
        micro_recall: 微查全率
        micro_f1: 微F1
        weighed_precision: 带权查准率
        weighed_recall: 带权查全率
        weighed_f1: 带权F1
    """
    from seaborn import heatmap

    fig, ax = plt.subplots(1, 2)
    subplot1 = heatmap(confusion_matrix, annot=True, ax=ax[0])
    subplot1.set_title("混淆矩阵热力图")

    row_sum = confusion_matrix.sum(axis=1, keepdims=True)
    col_sum = confusion_matrix.sum(axis=0, keepdims=True).ravel()
    all_sum = confusion_matrix.sum()

    norm_confusion_matrix = confusion_matrix / row_sum
    # 上面一行需要把每行的每一列除对应行的和，所以不能在最开始就降维（不然不同列就除的和就不一样了）
    row_sum = row_sum.ravel()

    np.fill_diagonal(norm_confusion_matrix, 0)
    subplot2 = heatmap(norm_confusion_matrix, annot=True, ax=ax[1])
    subplot2.set_title("错误率热力图")
    plt.show()

    true_positive = np.diagonal(confusion_matrix)
    false_positive = col_sum - true_positive
    true_negative = [
        true_positive.sum() - true_positive[i] for i in range(len(true_positive))
    ]

    recall = true_positive / row_sum
    precision = true_positive / col_sum
    false_positive_rate = false_positive / (false_positive + true_negative)
    accuracy = true_positive.sum() / all_sum

    f1_score = 2 / ((1 / precision) + (1 / recall))
    print("{:-^45}".format("以下为各类的数量"))
    print(f"实际数量：{row_sum}")
    print(f"预测数量：{col_sum}")
    print("{:-^45}".format("以下为以各类分别作为正例时的各项指标"))

    print(f"查准率(精度)为\n{precision}\n")
    print(f"查全率(真正例率/召回率)为\n{recall}\n")
    print(f"假正例率为\n{false_positive_rate}\n")
    print(f"F1分数为\n{f1_score}\n")
    print(f"准确率为\n{accuracy}\n")

    print("{:-^45}".format("以下为综合指标"))
    macro_precision = sum(precision) / len(precision)
    macro_recall = sum(recall) / len(recall)
    macro_f1 = (2 * macro_precision * macro_recall) / (macro_precision + macro_recall)

    micro_precision = sum(true_positive) / (sum(true_positive) + sum(false_positive))
    micro_recall = sum(true_positive) / sum(row_sum)
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

    weigh = np.array([row_sum / all_sum]).ravel()
    weighed_precision = sum(weigh * precision)
    weighed_recall = sum(weigh * recall)
    weighed_f1 = (2 * weighed_precision * weighed_recall) / (
        weighed_precision + weighed_recall
    )
    print(f"宏查准率为                      {macro_precision}")
    print(f"宏查全率为                      {macro_recall}")
    print(f"宏F1为                          {macro_f1}\n")
    print(f"微查准率为                      {micro_precision}")
    print(f"微查全率为                      {micro_recall}")
    print(f"微F1为                          {micro_f1}\n")
    print(f"加权查准率为                    {weighed_precision}")
    print(f"加权查全率为                    {weighed_recall}")
    print(f"加权F1为                        {weighed_f1}")

    return (
        precision,
        recall,
        false_positive_rate,
        f1_score,
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
        micro_precision,
        micro_recall,
        micro_f1,
        weighed_precision,
        weighed_recall,
        weighed_f1,
    )
