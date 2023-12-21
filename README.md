# modeltoolbox
最近发现这玩意导包的速度实在是太慢了，居然有四秒，正好跟matlab无关的函数写的越来越多了，把代
码重写一遍，顺便改个名字。  
省去无关文件后主程序目录树结构如下：  
.  
├── LICENSE.md  
├── modeltoolbox  
│   ├── __init__.py  
│   ├── main.py  
│   ├── mathon.py  
│   └── tools.py  
├── README.md  
├── setup.py  
└── share_engine.m  
因为matlab那个包和sns、sklearn的导入速度太慢，所以把软件全都模块化了，跟matlab有关的全放在
mathon.py里，目前对sns之类的库的处理是在函数体里面导入，缺点就是函数的加载变慢了，所以考虑
后期如果这些依赖用的足够多的话，就像matlab一样也单独成一个模块。  
  
也是为了提高导入速度，init.py中没有直接导入模块，所以如果要用到mathon和tools模块，只能通过
`import modeltoolbox.mathon as mmp`的方法导入。
  
`main`：主模块的函数  
`mathon`：原mathon中matlab相关函数的模块  
`tools`：主要是一些附加的数学函数和设置相关的模块

## main
以下是目前本模块所有函数的简介
### `preview(func)`
装饰器，用来运行plt.show()

### `save_plt_fig(path, hold)`
保存图片的函数，默认放在当前文件目录下的pypic文件夹中，可以手动选定路径，保存的图片依照先后顺序按数字从小到大命名。默认画完当前图像后自动hold off，如果需要hold on，给hold传入'on'就可以了。<br/>
用法：

```python
save_plt_fig(path='./pic, hold='off)
```
### `SavePreview_plt_fig():`
跟上面的函数用法完全相同，保存以后会自动跳出函数图像的图像，我觉得这里没办法设置hold on，但是还是写了一个判断传入的hold是on还是off的功能。

### `grid_caculator_multiprocessing(x, y, calculator, title='pic', xlabel='x', ylabel='y', zlabel='z',n_jobs=None,draw_pic=True):`
这里只是把mathon中的本函数和matlab有关的功能去掉以后照搬过来的，还没有作进一步的优化（下次一定）  
本函数用作并行计算复杂的图像，支持简单的字符串格式的简单函数关系，也支持复杂逻辑的运算<br/>
  
函数返回计算后的函数值，需要画图的话还需要提供x和y（可能还需要网格化），绘图的选项现在还没写，具体方法参考下面mathon中的内容。
  
n_jobs是并行数，不指定的话默认是cpu最大逻辑核心数。mathon中的函数也有这个选项，忘写了，我也懒得加了哈哈。  
用法这里放两个：<br/>首先是简单的函数关系的例子
```python
x = np.arange(-7, 7.1, 0.001)
y = np.arange(-8, 8.1, 0.001)
eng = mp.connect_matlab()
result = mp.mesh_multiprocessing(eng, x, y ,' x * np.exp(-x * 2 - y**2)')
```
<br/>
第二个是复杂逻辑运算的例子，就是数维杯内次的第二问嘛<br/>

```python
def func(C, a1, q):  # C是初始污垢量
    C0 = C
    k = 0
    a = a1
    while True:
        k += 1
        tmp = C * a  # 当次洗掉的污垢的量
        C = C - tmp
        a = a * q
        # if C/C0 <=0.001:  #  阈值
        if C / C0 <= 0.001:  #  阈值
            # print("衣服洗完了")
            # print(f"这是第{k}次")
            return k
        if k >= 1000:
            return -1


def suan(x, y):
    z = np.zeros((len(x), len(x[0])))
    for i in range(len(x)):
        for j in range(len(x[0])):
            z[i][j] = func(21, x[i][j], y[i][j])
    return z


x = np.linspace(0.01, 0.99, 500)
y = x.copy()

mtb.grid_caculator_multiprocessing(x, y, suan)
```

上面这段代码有优化的空间，现在他利用在global作用域中添加变量的方式来进行并行运算，目前我尝试过两种方法，第一种是创建闭包的计算函数，但是进程池的map方法不支持这样。第二种是通过手动和创建多个进程，并且用multiprocessing库里的队列来管理这几个进程，但是并没有达到管理的效果，最后的结果能算是能算，但是各个进程放入结果的顺序跟启动顺序不一样，最后出来的图片也是x轴上的排列顺序是混乱的。<br/>
另外提一句，这个函数提供了一个新的参数`useMatlab`这个参数表示是否利用matlab进行绘图，因为我测试了好多次，发现进程数一直是1，后来才发现那是因为cpu几秒钟就把结果算出来了，是matlab图画的太慢了，他matlab画图居然是单核运行的，已经跟不上python的速度了，所以如果让`useMatlab=False`，代码会返回计算好的结果，放到第一个例子里面就是每一个x和y所对应所有函数值的集合，后续如果能找到效率更高的绘图方法可以在此基础上改进一下。<br/>

### `calculate():`
上面`mesh_multiprocessing()`的辅助函数  

### `change_col_dtype(DataFrame, before, after)`
把DataFrame中的所有before类型的列转化为after类型。  
用法：
```python
DataFrame = mp.change_col_dtype(DataFrame, bool, int)
```

### `corr_heatmap(DataFrame, title='pic')`
快速绘制DataFrame中所有数字列的相关系数热力图（包括布尔列）

### `sklearn_model_report(model, train_data, test_data, scoring='accuracy')`
本函数用于输出已经训练好的**sklearn模块中的模型**的各项性能参数，scoring是某项或某些项参数
的名称，用于输出那些参数的平均估计值  
注意一下train_data是不含结果的数据集，test_data是前者的结果列而非测试集  

用法：  
```python
sklearn_model_report(clv_SVM, train_X, train_Y)
```



## mathon
以下是目前本模块所有函数的简介
目前所有使用matlab绘图的函数都实现了可以自定义标题和轴标签的功能，关键字参数名为：title,xlabel,ylabel

### `connect_matlab()`
本函数用于接收matlab共享的引擎接口，若不共享可以使用matlab.engine.start_matlab()来创建新的接口<br />
用法：
```python
eng = mp.connect_matlab()
```
也可以传入元组  
这个函数的行为被改成连接最后一个共享的matlab接口了，因为linux的matlab做的实在是一坨，关掉一个matlab以后他共享的接口居然是关不掉的，而且这个破matlab还老是把opengl的设置改回去，搞得我好几次都要开两遍接口，真的烦

### `to_matlab(*args)`
本函数主要用作本库内函数自行调用。<br />
用法：
```python
x, y = to_matlab(x, y)
```
其中，x和y均为python数组
### `to_matlab_gpu(eng, *args)`
照抄的上面的函数，用法也类似只是在本库中的plot_gpu()函数中调用本函数的时候，明明传入了两个python，最后却只返回了一个matlab数组，具体是什么我没有传入matlab工作区查看，如果遇到什么不清不楚的错误需要着重注意一下<font color="red">这里</font>  
用法：
```python
x, y = to_matlab_gpu(eng, x, y)
```
也可以传入元组

### `to_python(*args)`
本函数把MATLAB数组转化为python数组（不含gpuArray）<br />
用法：
```python
x, y = to_python(x, y)
```
### `matlab_gpu_to_python(eng, *args)`
本函数把gpuArray转化为python数组<br/>
用法：
```python
x, y = matlab_gpu_to_python(eng, x, y)
```
### `to_workspace(eng, dic)`
本函数把在python中储存的变量传入matlab工作区，仅支持传入matlab数据（未测试过python数组能否自动处理），可能需要结合前面的函数使用。<br/>
注意：本函数是以字典的形式传入工作区，键是变量名，值是变量值。例如如果想把1传入matlab的工作区，并赋予其a的名称，则需要传入{'a':1}。这样在matlab的工作区中则会出现该a=1的变量。<br/>
用法：

```python
to_workspace(eng, {'a'}:1)
```
### `read_workspace(eng, *args)`
本函数用于读取matlab中工作区的变量传入python，使用时输入字符型的matlab变量名。目前本函数用的不多，也没怎么测试，如果输入matlab中不存在的变量名可能会出错。本函数返回一个字典，字典的意义参见上个函数。<br/>
用法：

```python
dic = read_workspace(eng, 'a', 'b')
```
### `add_workspace(eng, x, y, z=None)`
这个函数我记得就是为了给下面几个绘图的函数用而临时写的，就是为了在画图的时候可以把自变量和因变量传入matlab，所以用法不写了。

### `plot(eng, *args)`
以x、y、z的顺序传入两个或者三个变量，利用matlab绘制一个二维曲线或者三维曲线图。<br/>
用法：

```python
plot(eng, x, y)
```
### `plot_gpu(eng, *args)`
plot函数的gpu版本，这个函数有点离谱，明明跟上面的函数结构一样，转换数组类型的时候这个函数就要解包一下，上面内个就不用，原因不明，可能是因为我在一行里面把数据类型换了两次？用的时候可能会出错。<br/>
后面考虑把画图函数分成本体和装饰器什么的重写一遍。<br/>
用法：

```python
plot_gpu(eng, x, y)
```
### `mesh(eng, x, y, z)`
本函数画三维曲面的图像，支持两种使用方式。<br/>
<br/>
第一种是先在函数外用np完成自变量的网格化和因变量取值的计算，然后传入函数，就像这样。
```python
x = np.arange(-2, 2.1, 0.01)
y = np.arange(-3, 3.1, 0.01)

x, y = np.meshgrid(x, y)
z = x*np.exp(-x**2 - y**2)
mp.mesh(eng, x, y, z)
```
<br/>

第二种是只传入自变量的范围和因变量对自变量的函数（字符串类型，不带等号），就像这样。<br/>
```python
x = np.arange(-2, 2.1, 0.01)
y = np.arange(-3, 3.1, 0.01)

mp.mesh(eng, x, y, "x*np.exp(-x**2-y**2)")
```
### `mesh_gpu()`
mesh函数的gpu版本，使用方法并无二致

### `mult_plot_str(eng, xticks, args, legend=None)`
本函数用来绘制一张图上有多条曲线的图像。<br/>
用法：

```python
mp.mult_plot_str(eng,['a','b','c'], [[1,2,3], [2,3,6]], ['a','b'])
```
上面第一项是x轴各点的值，最后一项是每条线的标签

### `mesh_multiprocessing(eng, x, y, calculator, title='pic', xlabel='x', ylabel='y', zlabel='z', useMatlab=True):`
本函数用作并行计算复杂的图像，支持简单的字符串格式的简单函数关系，也支持复杂逻辑的运算<br/>
用法这里放两个：<br/>首先是简单的函数关系的例子
```python
x = np.arange(-7, 7.1, 0.001)
y = np.arange(-8, 8.1, 0.001)
eng = mp.connect_matlab()
mp.mesh_multiprocessing(eng, x, y ,' x * np.exp(-x * 2 - y**2)')
```
<br/>
第二个是复杂逻辑运算的例子，就是数维杯内次的第二问嘛<br/>

```python
def func(C, a1, q):  # C是初始污垢量
    C0 = C
    k = 0
    a = a1
    while True:
        k += 1
        tmp = C * a  # 当次洗掉的污垢的量
        C = C - tmp
        a = a * q
        # if C/C0 <=0.001:  #  阈值
        if C / C0 <= 0.001:  #  阈值
            # print("衣服洗完了")
            # print(f"这是第{k}次")
            return k
        if k >= 1000:
            return -1


def suan(x,y):
    z = np.zeros((len(x), len(x[0])))
    for i in range(len(x)):
        for j in range(len(x[0])):
            z[i][j] = func(21, x[i][j], y[i][j])
    return z

x = np.linspace(0.01,0.99,500)
y = x.copy()

eng = mp.connect_matlab()
mp.mesh_multiprocessing(eng, x, y ,suan)
eng.quit()
```

上面这段代码有优化的空间，现在他利用在global作用域中添加变量的方式来进行并行运算，目前我尝试过两种方法，第一种是创建闭包的计算函数，但是进程池的map方法不支持这样。第二种是通过手动和创建多个进程，并且用multiprocessing库里的队列来管理这几个进程，但是并没有达到管理的效果，最后的结果能算是能算，但是各个进程放入结果的顺序跟启动顺序不一样，最后出来的图片也是x轴上的排列顺序是混乱的。<br/>
另外提一句，这个函数提供了一个新的参数`useMatlab`这个参数表示是否利用matlab进行绘图，因为我测试了好多次，发现进程数一直是1，后来才发现那是因为cpu几秒钟就把结果算出来了，是matlab图画的太慢了，他matlab画图居然是单核运行的，已经跟不上python的速度了，所以如果让`useMatlab=False`，代码会返回计算好的结果，放到第一个例子里面就是每一个x和y所对应所有函数值的集合，后续如果能找到效率更高的绘图方法可以在此基础上改进一下。<br/>

## tools
### `timer`
装饰器，用来计算某个函数的运算时间。  
用法：
```python
@timer()
def func(x, y):
    return x + y
```
### printable(dataFrame)
d忘了大写了...懒得改了，反正这东西我就用过一次...
用它只是因为载终端里用这个打印表格比较好看，但是后来发现pandas可以设置的也很好看，就不用了。

### prefer_settings():
喜欢的一些设置：  
plt生成的图像用ubuntu里的一款自带的中文字体，所以这个函数用在windows和termux上不报错的可能性不大。  
pandas打印表格的格式设置（用jukit的时候记得把终端窗口调到二等分屏幕）  

### C(a,b)
求组合数，$C_a^b$，就是C(a,b)，直接把a和b换成数即可

### A(a,b)
全排列，跟上面一样
## 问题
mp.plot_gpu()函数里面to_matlab_gpu()的调用里，给后者传入args的参数需要解包。暂时照这个做了
妈的我这个函数是直接搬的mp.plot()为什么那个函数不用解包这个函数就得解包？？？？？？？？？？？？？<br/>
离谱<br/>
args确实是元组而且确实长度为2<br/>
<br/>
<br/>
优化mesh_multiprocessing函数，具体的优化方向上面写了

