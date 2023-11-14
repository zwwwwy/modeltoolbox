from typing import Union

import matlab.engine
import numpy as np
from numpy import ndarray
import ast
from typing import Tuple, Any
import matplotlib.pyplot as plt
import os

counts = 1


class MatlabNotFound(Exception):
    pass


def connect_matlab():
    """
    使用此函数前要在matlab中运行matlab.engine.shareEngine！！！只能连接一个matlab引擎。
    :return: 返回值惯用eng接收
    """
    x = matlab.engine.find_matlab()
    if not x:
        raise MatlabNotFound(
            "未找到现有的MATLAB接口，记得前往MATLAB运行share_engine.m文件。（她还没有被删对嘛？）\n或者改为运行eng = matlab.engine.start_matlab()命令"
        )
    else:
        (x,) = x
        return matlab.engine.connect_matlab(x)


def to_matlab(*args) -> Union[matlab.double, tuple[matlab.double, ...]]:
    """
    将列表或numpy数组转化为matlab中的double数组
    :param args:
    :return: 返回与输入的参数相同数量的MATLAB中的double数组构成的元组（若只接受了一个参数则直接返回转化后的元组）
    """
    lst = []
    for i in args:
        lst.append(matlab.double(i))
    if len(lst) == 1:
        return lst[0]
    return tuple(lst)


def to_python(*args) -> Union[ndarray, tuple[ndarray, ...]]:
    """
    将MATLAB中的double数组转化为numpy数组
    :param args:
    :return: 返回与输入的参数相同数量numpy数组构成的元组（若只接受了一个参数则直接返回转化后的结果）
    """
    lst = []
    for i in args:
        lst.append(np.array(i))
    if len(lst) == 1:
        return lst[0]
    return tuple(lst)


def to_workspace(eng: matlab.engine.matlabengine.MatlabEngine, dic: dict):
    """
    用于向matlab的工作区中传入数据，字典的键对应工作区中的变量名，值对应工作区中的变量值
    :param eng: 引擎接口
    :param dic: dict
    :return: 无返回值
    """
    if (len_dic := len(dic)) == 0:
        print("并未向工作区传入任何参数")
    else:
        keys = list(map(str, dic.keys()))
        values = list(dic.values())
        for i in range(len_dic):
            eng.workspace[keys[i]] = values[i]


def read_workspace(eng: matlab.engine.matlabengine.MatlabEngine, *args: str):
    """
    输入引擎接口跟数个变量名，返回一个字典，未检查写的对不对，未写检查变量名是否存在的代码
    :param eng: 引擎接口
    :param args: 数个变量名
    :return:
    """
    dic = dict()
    for i in args:
        dic[i] = eng.workspace[i]
    return dic


def add_workspace(eng, x, y, z=None):
    """
    本函数为本模块中绘图函数在matlab中的工作区记录用
    :param eng:
    :param x:
    :param y:
    :param z:
    :return:
    """
    eng.workspace["x"] = x
    eng.workspace["y"] = y
    if z:
        eng.workspace["z"] = z


def plot(eng: matlab.engine.matlabengine.MatlabEngine, *args):
    """
    快速出线图，第一个参数为eng，剩下的两或三个参数为二或三个变量
    :param eng: 引擎接口
    :param args:
    :return:
    """
    if not 2 <= (length := len(args)) <= 3:
        raise ValueError("输入的参数数量错误，应该先输入eng后输入二或三个变量")
    if length == 3:
        x, y, z = to_matlab(args)
        add_workspace(eng, x, y, z)
        eng.plot3(x, y, z)
    if length == 2:
        x, y = to_matlab(args)
        add_workspace(eng, x, y)
        eng.plot(x, y)
    eng.ylabel("y")
    eng.grid("on", nargout=0)
    eng.title("pic")
    eng.xlabel("x")


def mesh(
    eng: matlab.engine.matlabengine.MatlabEngine,
    x: np.ndarray,
    y: np.ndarray,
    z: str = None,
):
    """
    输入参数应该为eng,x,y,z。其中，z的格式要求为等号左边只留下z，等号右边作为参数输入\n
    解析的所有公式中的sin、exp等数学函数需要带np.前缀！！！！（导入math不知道为什么会报错）
    :param eng: 引擎接口
    :param x: ndarray
    :param y: ndarray
    :param z: str
    :return: matlab图像
    """
    if not z:
        print("输入的参数数量错误，应该先输入eng后输入三个变量")
    x, y = np.meshgrid(x, y)

    # code = parser.expr(z).compile()  # 解析数学公式
    code = ast.parse(z, mode='eval')  # parser是标准库的函数,但是在3.11中被删掉了,用这个来代替
    z = eval(code)  # 计算z的值

    x, y, z = to_matlab(x, y, z)

    add_workspace(eng, x, y, z)

    eng.mesh(x, y, z)
    eng.ylabel("y")
    eng.grid("on", nargout=0)
    eng.title("pic")
    eng.xlabel("x")


def mult_plot_str(eng, xticks, args, legend=None):
    """
    在同一组坐标下绘制多个折线图，需要提供横坐标的值,在最后输入由图例名称组成的列表作为关键字参数legend
    :param eng: 引擎接口
    :param xticks: 横坐标
    :param args: 绘制的所有折线图的值的列表
    :param legend: 图例
    :return:
    """
    length = len(xticks)
    x = [i for i in range(1, length + 1)]
    x = matlab.double(x)  # 这里有点离谱，x是[[1,2,3,...]]这个库确实不好排除错误，matlab的语法真的是稀烂
    for i in args:
        plot(eng, x[0], i)
        eng.hold("on", nargout=0)
    if legend is not None:
        eng.legend(legend)
    eng.xticks(x, nargout=0)
    eng.xticklabels(xticks, nargout=0)
    eng.hold("off", nargout=0)


def preview(func):
    def inner(*args, **kwargs):
        func(*args, **kwargs)
        plt.show()

    return inner


def save_plt_fig(path="E:\\图片\\pypic", hold="off"):
    """
    保存plt中的图片，按数字顺序为图片命名，默认保存到E:\图片\pypic，可手动指定保存路径（绝对路径！！）
    另：程序默认画完一张图自动hold off，如需设置hold on需要给hold传入参数"on"
    太困了，没写自定义尺寸的参数，等以后有机会再写
    """
    global counts
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(f"{path}\\{counts}.jpg")
    counts += 1
    if hold == "on":
        pass
    else:
        plt.close()


@preview
def SavePreview_plt_fig(*args, **kwargs):
    """
    本函数所有参数都和save_plt_fig相同，详情参见前者
    在save_plt_fig的基础上利用装饰器实现在保存后展示图片，plt的默认设置好像是执行plt.show()
    以后自动hold=off，所以这里处理关键字hold没什么意义，在plt那里最后的结果都跟hold=off一样的，
    但是写都写了，再说万一以后有别的解决办法呢？就不删了
    """
    track = 1
    if "hold" in kwargs.keys():
        if kwargs["hold"] == "on":
            track = 0
        kwargs["hold"] = "on"
    save_plt_fig(*args, **kwargs)
    if track == 1:
        plt.close()

