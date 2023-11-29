from typing import Union

import matlab.engine
import numpy as np
from numpy import ndarray
import ast
from typing import Tuple, Any
import matplotlib.pyplot as plt
import os
import threading
from contextlib import contextmanager

_local = threading.local()
counts = 1


class MatlabNotFound(Exception):
    pass


def connect_matlab():
    """
    使用此函数前要在matlab中运行matlab.engine.shareEngine！！！只能连接一个matlab引擎。
    """
    x = matlab.engine.find_matlab()
    if not x:
        raise MatlabNotFound(
            "未找到现有的MATLAB接口，记得前往MATLAB运行share_engine.m文件。（她还没有被删对嘛？）\n或者改为运行eng = matlab.engine.start_matlab()命令"
        )
    else:
        x = x[-1]
        return matlab.engine.connect_matlab(x)


def to_matlab(*args) -> Union[matlab.double, tuple[matlab.double, ...]]:
    """ 将列表或numpy数组转化为matlab中的double数组

    Args:
        args:

    Returns:
        Union[matlab.double, tuple[matlab.double, ...]]:
    """
    lst = []
    for i in args:
        lst.append(matlab.double(i))
    if len(lst) == 1:
        return lst[0]
    return tuple(lst)


def to_matlab_gpu(eng, *args):
    """把一串数组转换为MATLAb的gpuArray，直接照抄的to_matlab

    Args:
        eng:
        args:
    """
    lst = []
    for i in args:
        lst.append(eng.gpuArray(matlab.double(i)))
    if len(lst) == 1:
        return lst[0]
    return tuple(lst)


def to_python(*args) -> Union[ndarray, tuple[ndarray, ...]]:
    """将MATLAB中的double数组转化为numpy数组

    Args:
        args:

    Returns:
        Union[ndarray, tuple[ndarray, ...]]:
    """
    lst = []
    for i in args:
        lst.append(np.array(i))
    if len(lst) == 1:
        return lst[0]
    return tuple(lst)


def matlab_gpu_to_python(eng, *args):
    """把MATLAB中的gpuArray转化为np数组，照抄to_python。目前不知道有什么用，但是以防万一先写一个
    未经测试，可能有bug！！！

    Args:
        eng:
        args:
    """
    lst = []
    for i in args:
        lst.append(np.array(eng.gather(i)))
    if len(lst) == 1:
        return lst[0]
    return tuple(lst)


def to_workspace(eng: matlab.engine.matlabengine.MatlabEngine, dic: dict):
    """用于向matlab的工作区中传入数据，字典的键对应工作区中的变量名，值对应工作区中的变量值

    Args:
        eng (matlab.engine.matlabengine.MatlabEngine): eng
        dic (dict): dic
    """
    if (len_dic := len(dic)) == 0:
        print("并未向工作区传入任何参数")
    else:
        keys = list(map(str, dic.keys()))
        values = list(dic.values())
        for i in range(len_dic):
            eng.workspace[keys[i]] = values[i]


def read_workspace(eng: matlab.engine.matlabengine.MatlabEngine, *args: str):
    """输入引擎接口跟数个变量名，返回一个字典，未检查写的对不对，未写检查变量名是否存在的代码

    Args:
        eng (matlab.engine.matlabengine.MatlabEngine): eng
        args (str): args
    """
    dic = dict()
    for i in args:
        dic[i] = eng.workspace[i]
    return dic


def add_workspace(eng, x, y, z=None):
    """本函数为本模块中绘图函数在matlab中的工作区记录用

    Args:
        eng:
        x:
        y:
        z:
    """
    eng.workspace["x"] = x
    eng.workspace["y"] = y
    if z:
        eng.workspace["z"] = z


def plot(eng: matlab.engine.matlabengine.MatlabEngine, *args):
    """快速出线图，第一个参数为eng，剩下的两或三个参数为二或三个变量

    Args:
        eng (matlab.engine.matlabengine.MatlabEngine): eng
        args:
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
    """输入参数应该为eng,x,y,z。其中，z的格式要求为等号左边只留下z，等号右边作为参数输入\n
    解析的所有公式中的sin、exp等数学函数需要带np.前缀！！！！（导入math不知道为什么会报错）

    Args:
        eng (matlab.engine.matlabengine.MatlabEngine): eng
        x (np.ndarray): x
        y (np.ndarray): y
        z (str): z
    """
    if not isinstance(z, str):
        x, y, z = to_matlab(x, y, z)
    if not z:
        print("输入的参数数量错误，应该先输入eng后输入三个变量")
        return
        x, y = np.meshgrid(x, y)

        # 以下三行再python3.11中能用，原来的解析函数被删掉了，不知道这三个在老版py里能不能用
        parsed_tree = ast.parse(z, mode='eval')
        compiled = compile(parsed_tree, filename='<string>', mode='eval')
        z = eval(compiled)
        x, y, z = to_matlab(x, y, z)

    add_workspace(eng, x, y, z)
    eng.mesh(x, y, z)
    eng.ylabel("y")
    eng.grid("on", nargout=0)
    eng.title("pic")
    eng.xlabel("x")


def mult_plot_str(eng, xticks, args, legend=None):
    """在同一组坐标下绘制多个折线图，需要提供横坐标的值,在最后输入由图例名称组成的列表作为关键字参数legend

    Args:
        eng: 引擎接口
        xticks: 横坐标
        args: 绘制的所有折线图的值的列表
        legend: 图例
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


def plot_gpu(eng, *args):
    """本函数利用MATLAB的gpuArray，使用gpu储存数组并进行数组的运算，然后将结果传入cpu进行图形的绘制

    Args:
        eng:
        args:
    """
    if not 2 <= (length := len(args)) <= 3:
        raise ValueError("输入的参数数量错误，应该先输入eng后输入二或三个变量")
    if length == 3:
        x, y, z = to_matlab_gpu(eng, args)
        add_workspace(eng, x, y, z)
        eng.plot3(x, y, z)
    if length == 2:
        x, y = to_matlab_gpu(
            eng, *args)  # 这里的args需要解包，为啥to_matlab不需要解包，to_matlab_gpu就得解包？？？？？
        # args是元组，长度为2，eng是引擎类型，也没问题，为什么要解包？？？？？先解一个再说把
        add_workspace(eng, x, y)
        eng.plot(x, y)
    eng.ylabel("y")
    eng.grid("on", nargout=0)
    eng.title("pic")
    eng.xlabel("x")


def mesh_gpu(
    eng: matlab.engine.matlabengine.MatlabEngine,
    x: np.ndarray,
    y: np.ndarray,
    z: str = None,
):
    """完全照抄mesh，就是把数据改成gpuArray,离谱，这里就不用解包？？？？？？？？？？？？？？？？？？
    
    Args:
        eng (matlab.engine.matlabengine.MatlabEngine): eng
        x (np.ndarray): x
        y (np.ndarray): y
        z (str): z
    """
    if not isinstance(z, str):
        x, y, z = to_matlab_gpu(x, y, z)
    if not z:
        print("输入的参数数量错误，应该先输入eng后输入三个变量")
        return
        x, y = np.meshgrid(x, y)

        # 以下三行再python3.11中能用，原来的解析函数被删掉了，不知道这三个在老版py里能不能用
        parsed_tree = ast.parse(z, mode='eval')
        compiled = compile(parsed_tree, filename='<string>', mode='eval')
        z = eval(compiled)
        x, y, z = to_matlab_gpu(x, y, z)

    add_workspace(eng, x, y, z)
    eng.mesh(x, y, z)
    eng.ylabel("y")
    eng.grid("on", nargout=0)
    eng.title("pic")
    eng.xlabel("x")


def mult_plot_str_gpu(eng, xticks, args, legend=None):
    """在同一组坐标下绘制多个折线图，需要提供横坐标的值,在最后输入由图例名称组成的列表作为关键字参数legend
    完全照抄mult_plot_str

    Args:
        eng:
        xticks:
        args:
        legend:
    """
    length = len(xticks)
    x = [i for i in range(1, length + 1)]
    x = matlab.double(x)  # 这里有点离谱，x是[[1,2,3,...]]这个库确实不好排除错误，matlab的语法真的是稀烂
    for i in args:
        plot_gpu(eng, x[0], i)
        eng.hold("on", nargout=0)
    if legend is not None:
        eng.legend(legend)
    eng.xticks(x, nargout=0)
    eng.xticklabels(xticks, nargout=0)
    eng.hold("off", nargout=0)


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
