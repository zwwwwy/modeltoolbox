import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# 这里先把plotly的go函数移除，等以后如果用这个接口多了再放到前面
# 多进程库同上，seaborn导入太慢，所以不在全局域导入


cpu_nums = os.cpu_count()


class Save_plt_fig:
    """Save_plt_fig.保存plt中的图片，按数字顺序为图片命名，默认保存到E:\\图片\\pypic，可手动指定保存路径（绝对路径！！）
    另：程序默认画完一张图自动hold off，如需设置hold on需要给hold传入参数"on"
    """

    def __init__(self, path="./pypic", hold="off"):
        self.path = path
        self.hold = hold
        self.counts = 1

    def __call__(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        print(self.counts)
        plt.savefig(f"{self.path}/{self.counts}.jpg")
        self.counts += 1
        if self.hold == "on":
            pass
        else:
            plt.close()


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
    import plotly.graph_objects as go
    import multiprocessing

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
        # stack = np.hstack(p.map(calculate, mpmesh_lsts))
        unstack = list(tqdm(p.imap(calculate, mpmesh_lsts), total=n_jobs))
        # 这里的进度条只是显示各个进程完成的情况，不精确

    stack = np.hstack(unstack)

    if draw_pic:
        fig = go.Figure(data=[go.Surface(x=x, y=y, z=stack)])

        # fig.update_layout(
        #     title=title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel  # 标题
        # )
        # 以上三行中的xlabel等参数在新版的plotly用不了了，所以改成了下面的
        fig.update_layout(title=title)

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


def corr_heatmap(DataFrame, title="pic", figsize=None, save_path=None):
    """heatmap.快速绘制出一个含有数字的DataFrame的相关系数热力图

    Args:
        DataFrame: pd.DataFrame
        title:
    """
    from seaborn import heatmap

    DataFrame = change_col_dtype(DataFrame, bool, int)
    numeric_columns = DataFrame.select_dtypes(include=["number"])
    corr = numeric_columns.corr()
    if figsize is not None:
        plt.figure(figsize=figsize)
    heatmap(corr, annot=True)
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    return corr


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
        print(
            f"在给出的训练集上，本模型{scoring}指标的多次预测平均值为：{_train.mean()}"
        )


def confusion_matrix_analysis(
    confusion_matrix, title1="混淆矩阵热力图", title2="错误率热力图"
):
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
    subplot1.set_title(title1)

    row_sum = confusion_matrix.sum(axis=1, keepdims=True)
    col_sum = confusion_matrix.sum(axis=0, keepdims=True).ravel()
    all_sum = confusion_matrix.sum()

    norm_confusion_matrix = confusion_matrix / row_sum
    # 上面一行需要把每行的每一列除对应行的和，所以不能在最开始就降维（不然不同列就除的和就不一样了）
    row_sum = row_sum.ravel()

    np.fill_diagonal(norm_confusion_matrix, 0)
    subplot2 = heatmap(norm_confusion_matrix, annot=True, ax=ax[1])
    subplot2.set_title(title2)
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


def learning_curve(model, x, y, title="学习曲线"):
    """learning_curve. 简单的学习曲线函数，sklearn的函数具体还不知道是个什么情况

    Args:
        model: 未训练好的模型
        x:
        y:
        title:
    Returns:
        model: 训练好的模型
    """
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
    train_errors, test_errors = [], []
    for m in range(1, len(train_x)):
        model.fit(train_x[:m], train_y[:m])
        train_y_predict = model.predict(train_x[:m])
        test_y_predict = model.predict(test_x)
        train_errors.append(mean_squared_error(train_y[:m], train_y_predict))
        test_errors.append(mean_squared_error(test_y, test_y_predict))
    RMSE_train = np.sqrt(train_errors)
    RMSE_test = np.sqrt(test_errors)
    plt.plot(RMSE_train, label="训练集")
    plt.plot(RMSE_test, "-+", label="测试集")
    plt.xlabel("训练集大小")
    plt.ylabel("误差(RMSE)")
    plt.title(title)
    plt.legend()
    plt.show()
    if len(test_errors) >= 10:
        delta = np.abs(RMSE_train[-10:] - RMSE_test[-10:])
        delta = 2 * delta / (RMSE_train[-10:] + RMSE_test[-10:])
        delta = delta.mean()
        if delta <= 0.1:
            print(
                f"训练集和测试集的误差相差{delta*100}%, 结果不一定好，注意一下误差，可能会过拟合"
            )
        else:
            print(f"训练集和测试集的误差相差{delta*100}%, 可能是欠拟合")
    return model


class Grey_model_11:
    def __init__(self):
        self.c = 0
        self.level = np.array([])  # 级比
        self.x1 = np.array([])  # 累加序列
        self.Z = np.array([])  # Z序列
        self.Y = np.array([])  # Y序列
        self.B = np.array([])  # B矩阵
        self.x = np.array([])  # 原始序列
        print("{:-^55}".format("以下为各项指标"))

    def fit(self, x):
        """fit. 这个函数进行级比检验，为了跟sklearn习惯接轨，所以叫fit了。
        这个平移的c求得比较玄学，我不知道为什么这么取，但是管用。
        c是每次递归时（递归的同时x自加c）累加x的最小值。

        Args:
            x: 仅接收列表和一维np数组，目前不想导入pd库
        """
        if isinstance(x, list):
            self.x = np.array(x)
        elif isinstance(x, np.ndarray):
            self.x = x
        elif isinstance(x, pd.Series):
            self.x = x.values
        self.level = self.x[:-1] / self.x[1:]
        lenth = len(self.x)
        if self.level.min() >= np.exp(-2 / (lenth + 1)) and self.level.max() <= np.exp(
            2 / (lenth + 2)
        ):
            print(f"级比检验完成，符合要求，平移了{self.c}，即c = {self.c}")
        else:
            self.c += self.x.min()
            self.x += self.x.min()
            self.fit(self.x)

    def predict(self, n):
        """predict. 返回预测的结果

        Args:
            n: 想要预测的个数
        """
        self.x1 = []
        tmp = 0
        for i in self.x:
            tmp += i
            self.x1.append(tmp)
        self.x1 = np.array(self.x1)
        self.Z = -0.5 * (self.x1[1:] + self.x1[:-1])
        self.Y = self.x[1:].T
        self.B = np.c_[self.Z.T, np.ones(len(self.Z)).T]
        result = np.linalg.inv((self.B.T.dot(self.B))).dot(self.B.T).dot(self.Y)
        a = result[0]
        b = result[1]
        self.a = a
        self.b = b
        print(f"发展系数a = {a}\n灰作用量b = {b}\n")

        predict_x1 = []

        for i in range(len(self.x1) + n):
            predict_x1.append((self.x[0] - b / a) * np.exp(-a * i) + b / a)
        predict_x1 = np.array(predict_x1)
        verfify = predict_x1[: len(self.x1)].copy()
        predict = predict_x1[len(self.x1) - 1 :].copy()
        verfify_x0 = verfify[1:] - verfify[:-1] - self.c
        predict_x0 = predict[1:] - predict[:-1] - self.c

        print("{:-^55}".format("以下为检验报告"))

        delta = np.abs(self.x[1:] - self.c - verfify_x0) / (self.x[1:] - self.c)
        delta = np.r_[0, delta]
        if np.array(delta < 0.1).all():
            print("相对误差检验达到了较高的要求")
        elif np.array(delta < 0.2).all():
            print("相对误差检验达到了一般的要求")
        else:
            print("相对误差检验不合格")

        rho = 1 - ((1 - 0.5 * a) / (1 + 0.5 * a)) * self.level
        if np.array(np.abs(rho) < 0.1).all():
            print("级比偏差检验达到了较高的要求")
        elif np.array(np.abs(rho) < 0.2).all():
            print("级比偏差检验达到了一般的要求")
        else:
            print("级比偏差检验不合格")

        print_verfify_x0 = np.r_[self.x[0] - self.c, verfify_x0]

        report = pd.DataFrame()
        report["序号"] = np.arange(1, len(self.x) + 1)
        report["原始值"] = self.x - self.c
        report["预测值"] = print_verfify_x0
        report["残差"] = self.x - self.c - print_verfify_x0
        report["相对误差"] = delta
        report["级比偏差"] = np.r_[np.nan, rho]
        report.set_index("序号", inplace=True)
        report.loc["均值"] = report.mean(axis=0)
        self.report = report

        print()
        print(report)

        print("{:-^55}".format("以下为预测结果"))
        result = pd.DataFrame()
        result["预测结果"] = predict_x0
        result["序号"] = np.arange(len(self.x) + 1, len(self.x) + 1 + n)
        result.set_index("序号", inplace=True)
        print(result)
        self.Z = -self.Z
        return predict_x0

    def get_report(self):
        """get_report. 返回检验报告"""
        return self.report


def grey_model_21(x):
    if isinstance(x, list):
        x = np.array(x)
    elif isinstance(x, np.ndarray):
        x = x
    elif isinstance(x, pd.Series):
        x = x.values

    x1, a1x0 = [], []
    tmp = 0
    for i in x:
        tmp += i
        x1.append(tmp)
    x1 = np.array(x1)

    a1x0 = x[1:] - x[:-1]
    Z = -0.5 * (x1[1:] + x1[:-1])

    B = np.c_[-x[1:].T, Z.T, np.ones(len(Z)).T]
    Y = a1x0.T
    u = np.linalg.inv((B.T.dot(B))).dot(B.T).dot(Y)
    a1 = u[0]
    a2 = u[1]
    b = u[2]
    print(f"x0的1-AGO序列x1为{x1}")
    print(f"x0的1-IAGO序列a1x0为{a1x0}")
    print(f"均值生成序列为{x}")
    print(f"B=\n{B}")
    print(f"Y={Y}\n")
    print("白化方程为（求解微分方程的时候可直接作为mathmetica的参数）：")
    print("未提供边界条件")
    if a1 < 0 and a2 < 0:
        print(f"x''[t] - {-a1} * x'[t] - {-a2} * x[t] == {b}")
    elif a1 < 0 and a2 > 0:
        print(f"x''[t] - {-a1} * x'[t] + {a2} * x[t] == {b}")
    elif a1 > 0 and a2 < 0:
        print(f"x''[t] + {a1} * x'[t] - {-a2} * x[t] == {b}")
    else:
        print(f"x''[t] + {a1} * x'[t] + {a2}* x[t] == {b}")


class DGM_21:
    def __init__(self):
        self.a = 0
        self.b = 0
        self.B = np.array([])
        self.Y = np.array([])
        self.u = np.array([])
        self.x = np.array([])
        self.x1 = np.array([])
        self.a1x0 = np.array([])

    def fit(self, x):
        if isinstance(x, list):
            self.x = np.array(x)
        elif isinstance(x, np.ndarray):
            self.x = x
        elif isinstance(x, pd.Series):
            self.x = x.values

    def predict(self, n):
        self.x1 = []
        tmp = 0
        for i in self.x:
            tmp += i
            self.x1.append(tmp)
        self.x1 = np.array(self.x1)

        self.a1x0 = self.x[1:] - self.x[:-1]

        self.B = np.c_[-self.x[1:].T, np.ones(len(self.x) - 1)]
        self.Y = self.a1x0.T
        self.u = np.linalg.inv(self.B.T.dot(self.B)).dot(self.B.T).dot(self.Y)

        a = self.u[0]
        b = self.u[1]
        self.a = a
        self.b = b

        predict = [self.x1[0]]
        for t in range(1, len(self.x) + n):
            predict.append(
                (b / a**2 - self.x[0] / a) * np.exp(-a * t)
                + b * t / a
                + (1 + a) * self.x[0] / a
                - b / a**2
            )

        predict = np.array(predict)
        predict_x0 = predict[1:] - predict[:-1]
        predict_x0 = np.r_[predict[0], predict_x0]
        self.verfify = predict_x0[:-n]
        self.predict_x0 = predict_x0[-n:]

        residual = self.x - self.verfify
        delta = np.abs(residual) / (self.x)
        self.report = pd.DataFrame()
        self.report["序号"] = np.arange(1, len(self.x) + 1)
        self.report["原始值"] = self.x
        self.report["预测值"] = self.verfify
        self.report["残差"] = residual
        self.report["相对误差"] = delta
        self.report.set_index("序号", inplace=True)
        self.report.loc["均值"] = self.report.mean(axis=0)

        print("{:-^55}".format("以下为检测报告"))
        print(self.report)

        print("{:-^55}".format("以下为预测结果"))
        print(self.predict_x0)

    def get_report(self):
        return self.report


class Markov_predict:
    def __init__(self):
        self.p1 = pd.DataFrame()

    def fit(self, x):
        judger = 1
        if isinstance(x, list):
            if isinstance(x[0], str):
                self.x = ""
                for i in x:
                    self.x += i
                judger = 1
            self.x = np.array(x).flatten()
        elif isinstance(x, np.ndarray):
            self.x = x.flatten()
        elif isinstance(x, pd.Series):
            self.x = x.values.flatten()
        elif isinstance(x, str):
            self.x = x
            judger = 0

        if judger == 1:
            self.x = "".join(map(str, self.x))
            self.x = self.x.replace(" ", "")

        unique = sorted(list(set(self.x)))
        self.p1 = pd.DataFrame(
            np.zeros((len(unique), len(unique))), index=unique, columns=unique
        )

        for i in range(len(self.x) - 1):
            self.p1.loc[self.x[i], self.x[i + 1]] += 1
        self.p1 = self.p1 / self.p1.to_numpy().sum(axis=1, keepdims=True)

        print("一步状态转移矩阵的估计如下")
        print(self.p1)

    def martix(self, n):
        self.result_lst = []

        def dot(n):
            result = self.p1 if n == 1 else self.p1.dot(dot(n - 1))
            self.result_lst.append(result)
            return result

        result = dot(n)
        print(f"\n{n}步状态转移矩阵的估计如下：")
        print(result)
        return result


class fast_universal_svm:
    def __init__(self):
        self.model = None
        self.train_data = None
        self.train_label = None

    def fit(self, x, y, C, loss="hinge", degree=None, kernel=None, **kwargs):
        self.train_data = x
        self.train_label = y

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC, LinearSVC

        if degree is None and kernel is None:
            print(f"将进行普通线性SVM，C={C}，损失函数为{loss}")
            self.model = Pipeline(
                [
                    ("标准化", StandardScaler()),
                    ("线性SVM", LinearSVC(C=C, loss=loss, **kwargs)),
                ]
            )
            print("训练完成")

        elif degree is not None and (kernel is None or kernel == "poly"):
            print(f"将进行多项式SVM，C={C}，损失函数为{loss}，多项式次数为{degree}")
            self.model = Pipeline(
                [
                    ("标准化", StandardScaler()),
                    ("多项式SVM", SVC(C=C, kernel="poly", degree=degree, **kwargs)),
                ]
            )
            print("训练完成")

        elif kernel == "rbf" and degree is None:
            print(f"将进行高斯RBF核SVM，C={C}，损失函数为{loss}")
            self.model = Pipeline(
                [
                    ("标准化", StandardScaler()),
                    ("RBF核SVM", SVC(C=C, kernel="rbf", **kwargs)),
                ]
            )

        else:
            print("不支持的参数")

        self.model.fit(self.train_data, self.train_label)
        print("训练完成")
        return self.model

    def predict(self, x):
        result = self.model.predict(x)
        print(f"预测结果为: \n{result}")
        return result

    def get_report(self, test_data=None, test_label=None):
        from sklearn.metrics import confusion_matrix

        if test_data is None and test_label is None:
            pred = self.model.predict(self.train_data)
            conf = confusion_matrix(self.train_label, pred)
            print("在训练集上的分析结果如下：\n")
            result = confusion_matrix_analysis(conf)

        else:
            pred = self.model.predict(test_data)
            conf = confusion_matrix(test_label, pred)
            print("在测试集上的分析结果如下：\n")
            result = confusion_matrix_analysis(conf)

        return result


def evr_plot(data, cross_line_y=None, title="主成分方差贡献率图"):
    """evr_plot. 画出主成分方差贡献率图, 返回累计可解是方差

    Args:
        data:
        title:
    """
    from sklearn.decomposition import PCA

    pca = PCA()
    pca.fit(data)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    x = np.arange(1, len(cumsum) + 1)
    plt.plot(x, cumsum, label="累计可解释方差")
    plt.title(title)
    plt.xlabel("维度")
    plt.ylabel("可解释方差")
    if cross_line_y is not None:
        y = np.ones(len(x)) * cross_line_y
        plt.plot(x, y, "r--", label=f"y={cross_line_y}")
    plt.legend()
    plt.show()
    return cumsum


def plot_k_in_kmeans(data, begin=2, end=10):
    """compute_k_in_kmeans.
    本函数画出轮廓分数和k的关系图以及各种k值的轮廓图分析，用于确定聚类中的最佳k值
    默认从k=2试到k=10，后面可以根据图像进一步缩小范围再运行本函数
    本函数默认随机种子=42

    Args:
        data: 带聚类的数据
        begin: k的起始值
        end: k的终止值
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, silhouette_samples

    k_lst = np.arange(begin, end + 1)

    n_cycle = sum(k_lst)
    update_cycle = 100 / n_cycle

    n_row = int((end + 1) ** 0.5)
    n_col = (end - begin + 1) // n_row
    if n_row * n_col < end - begin + 1:
        n_col += 1
    fig, ax = plt.subplots(n_row, n_col, sharex="col")
    ax = ax.flatten()
    silhouette_score_val_lst = []
    with tqdm(total=100) as pbar:
        for i, k in enumerate(k_lst):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            # 上面的算法运行次数设置为auto是为了不报警告，看着烦，原来默认是10
            y_pred = kmeans.fit_predict(data)
            silhouette_score_val = silhouette_score(data, y_pred)
            silhouette_score_val_lst.append(silhouette_score_val)
            silhouette_samples_val = silhouette_samples(data, y_pred)
            y_lower = 10
            for j in range(
                k
            ):  # 遍历每一个集群，取第i个簇中对应所有样本的轮廓系数，并进行排序
                s_values = silhouette_samples_val[y_pred == j]
                s_values.sort()
                size_cluster_j = s_values.shape[0]  # 得到第j个簇的样本数量
                y_upper = y_lower + size_cluster_j  # 图中每个簇在y轴上的宽度
                ax[i].fill_betweenx(
                    y=np.arange(y_lower, y_upper), x1=0, x2=s_values, alpha=0.7
                )
                ax[i].text(
                    -0.05, y_lower + 0.5 * size_cluster_j, str(j)
                )  # 在y轴右侧标记每个簇的序号
                y_lower = y_upper + 10
                pbar.update(update_cycle)
            ax[i].axvline(
                x=silhouette_score_val, color="red", linestyle="--"
            )  # 画出轮廓系数的平均值的直线（垂直于x轴）

            if i % n_col == 0:
                ax[i].set_ylabel("集群")
            for tmp in ax[-n_col:]:
                tmp.set_xlabel("轮廓系数")
            ax[i].set_title(f"K={k}, 平均轮廓系数={silhouette_score_val:.2f}")
            ax[i].set_yticks([])
    plt.show()
    plt.plot(k_lst, silhouette_score_val_lst)
    plt.scatter(k_lst, silhouette_score_val_lst)
    plt.xticks(k_lst)
    plt.title("k与轮廓分数的关系")
    plt.xlabel("k")
    plt.ylabel("轮廓分数")
    plt.show()


def regression_report(predict, real):
    """regression_report.
    计算回归模型的各项指标
    Args:
        predict:
        real:
    """
    difference = real - predict
    length = len(real)
    mse = (difference**2).sum() / length
    rmse = mse**0.5
    mae = np.abs(difference).sum() / length
    mape = np.abs(difference / length).sum() / length
    evc = 1 - np.var(difference) / np.var(real)

    mean_y = np.mean(real)
    sst = np.sum((real - mean_y) ** 2)
    sse = np.sum((real - predict) ** 2)
    ssr = sst - sse
    r2 = 1 - sse / sst

    print(f"均方误差（MSE）为               {mse}")
    print(f"均方根误差（RMSE）为            {rmse}")
    print(f"平均绝对误差（MAE）为           {mae}")
    print(f"平均绝对百分比误差（MAPE）为    {mape}")
    print(f"可解释方差得分（EVC）为         {evc}")
    print(f"决定系数（R-square）为          {r2}")
    print(f"回归平方和SSR为                 {ssr}")
    print(f"残差平方和SSE为                 {sse}")
    print(f"总平方和SST为                   {sst}")

    return mse, rmse, mae, mape, evc, r2, ssr, sse, sst


def single_roc_pr_curve(model_name, classes, label, score):
    """
    画出ROC曲线和P-R曲线(每一类单独做图)
    Args:
        model_name: 模型名称, 如果有多个模型，传入列表，否则传入字符串
        classes: 类别
        label: 真实标签
        score: 分数(proba或decision_function)，如果有多个模型，传入所有模型的分数构成的数组
    """
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    from sklearn.preprocessing import label_binarize

    y = label_binarize(label, classes=classes)
    if (length := len(classes)) == 2:
        fig, axs = plt.subplots(1, 2)
    else:
        fig, axs = plt.subplots(length, 2)

    if isinstance(model_name, list):
        for j in range(len(model_name)):
            if len(classes) == 2:
                fpr, tpr, _ = roc_curve(y, score[j][:, 1])
                roc_auc = auc(fpr, tpr)
                precision, recall, _ = precision_recall_curve(y, score[j][:, 1])
                axs[0].plot(fpr, tpr, label=f"{model_name[j]},auc={roc_auc:0.3f}")
                axs[0].plot([0, 1], [0, 1], "b--")
                axs[0].set_title(f"ROC曲线, auc={roc_auc}")
                axs[0].set_xlabel("假正例率")
                axs[0].set_ylabel("真正例率")
                axs[1].plot(recall, precision, label=f"{model_name[j]}")
                axs[1].set_title("P-R曲线")
                axs[1].set_xlabel("召回率")
                axs[1].set_ylabel("查准率")
                axs[0].legend()
                axs[1].legend()
            else:
                for i in range(length):
                    fpr, tpr, _ = roc_curve(y[:, i], score[j][:, i])
                    roc_auc = auc(fpr, tpr)
                    precision, recall, _ = precision_recall_curve(
                        y[:, i], score[j][:, i]
                    )
                    axs[i, 0].plot(
                        fpr, tpr, label=f"{model_name[j]},auc={roc_auc:0.3f}"
                    )
                    axs[i, 0].plot([0, 1], [0, 1], "b--")
                    axs[i, 0].set_title(f"类别{classes[i]}的ROC曲线, auc={roc_auc}")
                    axs[i, 0].set_xlabel("假正例率")
                    axs[i, 0].set_ylabel("真正例率")
                    axs[i, 1].plot(recall, precision, label=f"{model_name[j]}")
                    axs[i, 1].set_title(f"类别{classes[i]}的P-R曲线")
                    axs[i, 1].set_xlabel("召回率")
                    axs[i, 1].set_ylabel("查准率")
                    axs[i, 0].legend()
                    axs[i, 1].legend()
        plt.tight_layout()
        plt.show()

    else:
        if len(classes) == 2:
            fpr, tpr, _ = roc_curve(y, score[:, 1])
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(y, score[:, 1])
            axs[0].plot(fpr, tpr, "r", label=f"{model_name},auc={roc_auc:0.3f}")
            axs[0].plot([0, 1], [0, 1], "b--")
            axs[0].set_title(f"ROC曲线, auc={roc_auc}")
            axs[0].set_xlabel("假正例率")
            axs[0].set_ylabel("真正例率")
            axs[1].plot(recall, precision, "orange", label=f"{model_name}")
            axs[1].set_title("P-R曲线")
            axs[1].set_xlabel("召回率")
            axs[1].set_ylabel("查准率")
            axs[0].legend()
            axs[1].legend()
        else:
            for i in range(length):
                fpr, tpr, _ = roc_curve(y[:, i], score[:, i])
                roc_auc = auc(fpr, tpr)
                precision, recall, _ = precision_recall_curve(y[:, i], score[:, i])
                axs[i, 0].plot(fpr, tpr, "r", label=f"auc={roc_auc:0.3f}")
                axs[i, 0].plot([0, 1], [0, 1], "b--")
                axs[i, 0].set_title(f"类别{classes[i]}的ROC曲线, auc={roc_auc}")
                axs[i, 0].set_xlabel("假正例率")
                axs[i, 0].set_ylabel("真正例率")
                axs[i, 0].legend()
                axs[i, 1].plot(recall, precision, "orange")
                axs[i, 1].set_title(f"类别{classes[i]}的P-R曲线")
                axs[i, 1].set_xlabel("召回率")
                axs[i, 1].set_ylabel("查准率")
        plt.tight_layout()
        plt.show()


def multi_roc_pr_curve(classes, label, score):
    """
    画出ROC曲线和P-R曲线(所有类别一起做图)
    Args:
        classes: 类别
        label: 真实标签
        score: 分数(proba或decision_function)
    """
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    from sklearn.preprocessing import label_binarize

    y = label_binarize(label, classes=classes)
    fig, axs = plt.subplots(1, 2)
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y[:, i], score[:, i])
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y[:, i], score[:, i])
        axs[0].plot(fpr, tpr, label=f"{classes[i]},auc={roc_auc:0.3f}")
        axs[0].plot([0, 1], [0, 1], "b--")
        axs[0].set_title("ROC曲线")
        axs[0].set_xlabel("假正例率")
        axs[0].set_ylabel("真正例率")
        axs[1].plot(recall, precision, label=f"{classes[i]}")
        axs[1].set_title("P-R曲线")
        axs[1].set_xlabel("召回率")
        axs[1].set_ylabel("查准率")
        axs[0].legend()
        axs[1].legend()

    plt.show()


def find_error(positive, pred, label, data):
    """
    找出数据集中所有的TP, FP, TN, FN（OvR）
    Args:
        positive: 正例
        pred: 预测值
        label: 真实值
        data: 数据
    """

    tp = data[(pred == positive) & (label == positive)]
    fp = data[(pred == positive) & (label != positive)]
    tn = data[(pred != positive) & (label != positive)]
    fn = data[(pred != positive) & (label == positive)]
    return tp, fp, tn, fn


class Auto_ARIMA:
    def __init__(self):
        # 初始化

        self.model = None
        self.isdiff = None
        self.d = None
        self.p = None
        self.q = None
        self.data = None
        self.alpha = 0.05
        self.diff_n = 0
        self.restore = []

    def fit(self, data, max_p=6, max_q=6):
        # from statsmodels.tsa.arima_model import ARIMA
        import statsmodels.api as sm
        from statsmodels.tsa.stattools import adfuller

        # 这里先进行ADF检验判断数据是否平稳，不平稳进行差分操作，记录差分阶数以备后续还原
        # 然后使用特定方法判断AR和MA的阶数
        # 返回预测好的原始模型，然后赋值给self.model
        self.data = data
        while True:
            adf_result = adfuller(self.data)
            if (
                adf_result[1] < self.alpha
            ):  # p_value值大，无法拒接原假设,有可能单位根，需要T检验
                print(f"差分阶数为{self.diff_n}，数据平稳")
                self.d = self.diff_n
                break
            else:
                if (
                    adf_result[0] < adf_result[4]["5%"]
                ):  # 代表t检验的值小于5%,置信度为95%以上，这里还有'1%'和'10%'
                    print(f"差分阶数为{self.diff_n}，数据平稳")
                    self.d = self.diff_n
                    break  # 拒接原假设，无单位根，平稳的
                else:
                    self.restore.append(self.data[0])  # 添加data的第一个值，用于还原
                    self.data = np.diff(self.data)  # 无法拒绝原假设，有单位根，不平稳的
                    self.diff_n += 1
                    if self.diff_n >= 1000:
                        print("差分阶数超过1000，无法平稳")
                        return

        # 使用aic确定p和q的阶数
        aic_values = {}
        for p in range(max_p):
            for q in range(max_q):
                model = sm.tsa.ARIMA(self.data, order=(p, self.d, q))
                try:
                    result = model.fit()
                except Exception as e:
                    print(f"ARIMA{p}{self.d}{q}模型不可用, error: {e}")
                    continue
                aic_values[(p, self.d, q)] = result.aic
        min_aic = min(aic_values, key=aic_values.get)
        print(f"最佳的ARIMA模型为ARIMA{min_aic}")
        self.p = min_aic[0]
        self.q = min_aic[2]
        self.model = sm.tsa.ARIMA(self.data, order=min_aic).fit()
        self.restore = self.restore[::-1]

    def predict(self, step):
        # 这里应该添加一个参数表示返回真实值还是差分值
        # 如果返回真实值，先判断是否差分过，若差分过则还原
        # 返回预测值
        result = self.model.forecast(step)
        if self.d:
            for i in range(len(self.restore)):
                result = np.r_[self.restore[i], self.data, result]
                result = np.cumsum(result)
            return result[-step:]
        else:
            return result

    def report(self):
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        print(self.model.summary())
        fig, axes = plt.subplots(3, 1, figsize=(10, 9))
        plot_acf(self.data, ax=axes[0])
        plot_pacf(self.data, ax=axes[1])
        residuals = pd.DataFrame(self.model.resid)
        residuals.plot(ax=axes[2])
        plt.title("Residuals")
        plt.tight_layout()
        plt.show()

    def test(self, start=0, end=-1, alpha=1):
        from statsmodels.graphics.tsaplots import plot_predict

        past_predictions = self.model.predict(start=start, end=end, dynamic=False)
        plot_predict(self.model, start=start, end=end, dynamic=False)
        if end == -1:
            plt.plot(self.data[start:], label="真实值", alpha=alpha)
        else:
            plt.plot(self.data[start : end + 1], label="真实值", alpha=alpha)
        plt.legend()
        plt.show()

        regression_report(past_predictions, self.data)
        print(past_predictions, self.data)

        # 返回预测的过去数据
        return past_predictions


class Lagrange_interpolation:
    def __init__(self):
        self.x = None
        self.y = None
        self.n = None
        self.m = None
        self.res = []

    def fit(self, x, y):
        """
        x: x坐标
        y: y坐标
        """
        self.x = x
        self.y = y
        self.n = len(x)

    def predict(self, x):
        if not self.res:
            self.res = []
        if isinstance(x, int) or isinstance(x, float):
            self.m = 1
        else:
            self.m = len(x)
        for i in range(self.m):
            result = 0
            for j in range(self.n):
                tmp = 1
                for k in range(self.n):
                    if k != j:
                        tmp *= (x[i] - self.x[k]) / (self.x[j] - self.x[k])
                result += tmp * self.y[j]
            self.res.append(result)
        return self.res


class Liner_interpolation:
    def __init__(self):
        self.x = None
        self.y = None
        self.n = None
        self.m = None
        self.res = []

    def fit(self, x, y):
        """
        x: x坐标
        y: y坐标
        """
        self.x = x
        self.y = y
        self.n = len(x)

    def predict(self, x):
        if not self.res:
            self.res = []
        if isinstance(x, int) or isinstance(x, float):
            self.m = 1
        else:
            self.m = len(x)
        for i in range(self.m):
            for j in range(self.n - 1):
                if self.x[j] <= x[i] <= self.x[j + 1]:
                    result = (self.y[j + 1] - self.y[j]) / (
                        self.x[j + 1] - self.x[j]
                    ) * (x[i] - self.x[j]) + self.y[j]
                    self.res.append(result)
                    break
        return self.res


class Csape_interpolation:
    def __init__(self):
        self.x = None
        self.y = None
        self.n = None
        self.m = None
        self.res = []

    def fit(self, x, y):
        """
        x: x坐标
        y: y坐标
        """
        self.x = x
        self.y = y
        self.n = len(x)

    def predict(self, x):
        if not self.res:
            self.res = []
        if isinstance(x, int) or isinstance(x, float):
            self.m = 1
        else:
            self.m = len(x)
        from scipy.interpolate import CubicSpline

        cs = CubicSpline(self.x, self.y)
        for i in range(self.m):
            result = cs(x[i])
            self.res.append(result)
        return self.res


def chi2_test(data):
    """
    卡方检验
    Args:
        data: 数据
    """
    from scipy.stats import chi2_contingency

    chi2, p, dof, expected = chi2_contingency(data)
    print(f"卡方值为{chi2}")
    print(f"p值为{p}")
    print(f"自由度为{dof}")
    print(f"期望值为{expected}")
