import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def confusion_matrix_analysis(confusion_matrix, title1="混淆矩阵热力图", title2="错误率热力图"):
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
            print(f"训练集和测试集的误差相差{delta*100}%, 结果不一定好，注意一下误差，可能会过拟合")
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
