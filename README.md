# modeltoolbox
本库的主模块使用import mtb导入  
啊啊啊啊啊我吐了，写了一晚上评价混淆矩阵的函数，写完以后才发现sklearn有这个函数...我不管了我就要用我的，我写的功能也比他多...啊啊啊啊  
最近发现这玩意导包的速度实在是太慢了，居然有四秒，正好跟matlab无关的函数写的越来越多了，把代
码重写一遍，顺便改个名字。  
省去无关文件后主程序目录树结构如下：  

```txt
.
├── LICENSE.md
├── mtb
│   ├── __init__.py
│   ├── main.py
│   ├── mathon
│   │   ├── __init__.py
│   │   └── main.py
│   ├── tools
│   │   ├── __init__.py
│   │   └── main.py
│   └── uncomplete
│       └── uncomplete.py
├── README.md
├── setup.py
└── share_engine.m
```

因为matlab那个包和sns、sklearn的导入速度太慢，所以把软件全都模块化了，跟matlab有关的全放在
mathon.py里，目前对sns之类的库的处理是在函数体里面导入，缺点就是函数的加载变慢了，所以考虑
后期如果这些依赖用的足够多的话，就像matlab一样也单独成一个模块。  

也是为了提高导入速度，init.py中没有直接导入模块，所以如果要用到mathon和tools模块，只能通过
`import mtb.mathon as mp`的方法导入。

`main`：主模块的函数  
`mathon`：原mathon中matlab相关函数的模块  
`tools`：主要是一些附加的数学函数和设置相关的模块
`uncomplete`：未完成

## main
以下是目前本模块所有函数的简介

### `Save_plt_fig(path, hold)`
保存图片的函数，默认放在当前文件目录下的pypic文件夹中，可以手动选定路径，保存的图片依照先后顺序按数字从小到大命名。默认画完当前图像后自动hold off，如果需要hold on，给hold传入'on'就可以了。<br/>
用法：

```python
a = Save_plt_fig(path='./pic, hold='off)

...(plt1)
a()

...(plt2)
a()
```

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


def operator(x, y):
    z = np.zeros((len(x), len(x[0])))
    for i in range(len(x)):
        for j in range(len(x[0])):
            z[i][j] = func(21, x[i][j], y[i][j])
    return z


x = np.linspace(0.01, 0.99, 500)
y = x.copy()

mtb.grid_caculator_multiprocessing(x, y, operator)
```
有个需要注意的地方，operator处理的x必须是按照并行数平均切分好的列表！！！！！  
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
用法：  

```python
mtb.corr_heatmap(df)
```

### `fast_corrscatter_evaluate(DataFrame, target, n=4)`
快速绘制DataFrame中和target项相关系数最高的n列（默认为4）的相关系数散点图。

```python
mtb.fast_corrscatter_evaluate(df, 'median_house_value')
```

### `sklearn_model_report(model, train_data, test_data, scoring='accuracy')`
本函数用于输出已经训练好的**sklearn模块中的模型**的各项性能参数，scoring是某项或某些项参数
的名称，用于输出那些参数的平均估计值  
注意一下train_data是不含结果的数据集，test_data是前者的结果列而非测试集  

用法：  
```python
sklearn_model_report(clv_SVM, train_X, train_Y)
```

### `confusion_matrix_analysis(confusion_matrix)`
本函数用于分析任意混淆矩阵  
用法：  

```python
conf_mx = np.array(
    [
        [85, 20],
        [15, 280],
    ]
)

(
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
) = confusion_matrix_analysis(conf_mx)
```
这个函数会输出两个图像，第一个是用混淆矩阵绘制的热力图，可以看到混淆矩阵的各项数值大小。第二个图像是错误率图像，这个图像是用混淆矩阵中每个元素分别除以对应行的和（实际为某类的数量），这个值的意义，举个例子，就是在所有的8中，错误的被预测为1，2，3...等其他类的分别的比率。因为第二个图像是研究错误率，所以把该图片对角线上的元素（预测正确的）都填充为0  
在输出两个图像以后，本函数还会计算混淆矩阵的各个参数，然后一次输出并返回这些参数，比如上面用法的输出如下：  

```txt
-----------------以下为各类的实际数量------------------
实际数量：[105 295]
预测数量：[100 300]
-------------以下为以各类分别作为正例时的各项指标--------------
查准率(精度)为
[0.85       0.93333333]

查全率(真正例率/召回率)为
[0.80952381 0.94915254]

假正例率为
[0.05084746 0.19047619]

F1分数为
[0.82926829 0.94117647]

准确率为
0.9125

-------------------以下为综合指标-------------------
宏查准率为                      0.8916666666666666
宏查全率为                      0.8793381759483454
宏F1为                          0.8854595101647088

微查准率为                      0.9125
微查全率为                      0.9125
微F1为                          0.9125

加权查准率为                    0.9114583333333334
加权查全率为                    0.9125000000000001
加权F1为                        0.9119788692175899

```

结合混淆矩阵的结构  
|                  | Predicted Positive | Predicted Negative |
|------------------|--------------------|--------------------|
| Actual Positive  | TP (True Positives)| FN (False Negatives)|
| Actual Negative  | FP (False Positives)| TN (True Negatives)|

注意到这里输出的各项参数均为两个，而常见的参数分析只有一个，这两个值的意义是，第一个值表示把第一类作为正例，其他类作为反类计算的结果，第二个表示把第二个视作正类，其他类视作反类的计算结果，以此类推。  
附上上面几个参数的计算公式:  
$Precision = \frac{TP}{TP+FP}$  

$Recall = \frac{TP}{TP+FN}$  

$FPR = \frac{FP}{FP+TN}$  

$F1=\frac{2\cdot P\cdot R}{P+R}$  

$macro\_P=\frac1n\sum\limits^n_{i=1}P_i$  

$macro\_R=\frac1n\sum\limits^n_{i=1}R_i$  

$macro\_F1=\frac{2\cdot macro\_P\cdot macro\_R}{macro\_P+macro\_R}$  

$micro\_P=\frac{\overline{TP}}{\overline{TP}+\overline{FP}}$  

$micro\_R=\frac{\overline{TP}}{\overline{TP}+\overline{FN}}$  

$micro\_F1=\frac{2\cdot micro\_P\cdot micro\_R}{micro\_P+micro\_R}$  

权重的式子就是把各个P啊R啊加权求和（权重就是各类实际数量占总样本数的比值）

### `Grey_model_11`
这是一个类，他是GM(1,1)模型，有fit、predict和get_report三个方法（不含初始化）  
#### `Grey_model_11.fit(self, x)`
叫这个名字是为了符合sklearn的习惯，本方法接收待预测的序列
#### `Grey_model_11.predict(self, n)`
本方法接收想要预测的个数，并自动进行结果检验并输出报告，返回预测结果(np.array)
#### `Grey_model_11.get_report(self)`
本方法无参数，直接返回检验报告(pd.DataFrame)  

用法:   
这里用的例子是司守奎的数学建模算法与应用(matlab)的（数学式子也是）

```python
gm11_pred = mtb.Grey_model_11()
gm11_pred.fit([71.1, 72.4, 72.4, 72.1, 71.4, 72.0, 71.6])
result = gm11_pred.predict(5)
```

### `grey_model_21`
这里只写到把白化方程和所有参数求出的地方，具体求解白化方程的地方我放到uncomplete里回头再说了，我不会求解微分方程所以到时候放到mathematica手动解了。  
用法不写了，参数就是所有数据的一行列表

### `DGM_21`
类，具体的形式跟上面GM(1,1)一样的，不写了

### `Markov_predict`

类，主要有俩方法，一个是fit一个是matrix

用法: 

```python
m = Markov_predict()
m.fit(
    [
        [4, 3, 2, 1, 4, 3, 1, 1, 2, 3],
        [2, 1, 2, 3, 4, 4, 3, 3, 1, 1],
        [1, 3, 3, 2, 1, 2, 2, 2, 4, 4],
        [2, 3, 2, 3, 1, 1, 2, 4, 3, 1],
    ]
)
print(m.martix(4))
```

此外，如果已经知道一步转移矩阵，也可以不用fit方法，比如  

```python
m = Markov_predict()
m.p1 = np.array(
    [
        [0.8, 0.1, 0.1],
        [0.5, 0.1, 0.4],
        [0.5, 0.3, 0.2],
    ]
)
print(np.array([0.2, 0.4, 0.4]).dot(m.martix(3)))
```

fit是生成一步状态转移矩阵，matrix是接收参数n输出n步状态转移矩阵，懒得写太多了。

注意一下我加了一个特殊属性是result_lst，是n步前的所有状态转移矩阵的列表，可以用`result_lst = m.result_lst`来获取

### `fast_universal_svm`

类，快速训练及评估常见的SVM模型（线性，多项式内核，高斯RBF内核。

#### `fit`

训练集和模型里的所有参数都放在这里，本函数返回训练好的模型

#### `predict`

接收被预测的数据

####  `get_report`

如果有参数（测试集）则按在测试集上的混淆矩阵来评估。

如果没参数，就按在训练集上的混淆矩阵来评估

### `evr_plot(data, cross_line_y=None, title=...)`
本函数绘制主成分方差的贡献率图，在本图中可以看出数据降到多少维的时候仍然数据仍保留较大的差异  
cross_line_y参数是绘制一条等于这个值的平行与y轴的直线

用法：  

```python
from sklearn.datasets import fetch_openml
import mtb

mnist = fetch_openml("mnist_784", version=1)
mtb.evr_plot(mnist["data"], 0.95)
```

这里还划了一条y=0.95的虚线，表示在保留95%的差异的化，维度大致是多少

### `plot_k_in_kmeans`
本函数用于估计k-means聚类的k值，可以重复多次运行，介绍写文档里了，懒得抄了。  
用法：  

```py
from sklearn.datasets import make_blobs
from mtb.tools import prefer_settings

prefer_settings()

X, y = make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,
    cluster_std=1,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=1,
)

plot_k_in_kmeans(X)
```

### `regression_report(predict, real):`
predict是真实值，real是预测值，输出各个评价参数，下附公式  
$MSE=\sum\limits^n_{i=1}(y_i-\widehat{y_i})^2\cdot\frac1n$  
  
$RMSE=\sqrt{MSE}$  
  
$MAE=\sum\limits^n_{i=1}|y_i-\widehat{y_i}|\cdot\frac1n$  
  
$MAPE=\sum\limits^n_{i=1}|\frac{y_i-\widehat{y_i}}{n}|\cdot\frac1n$  
  
$R-square=1-\frac{MSE}{Var(y_{real})}$  
  
$EVC=\frac{Var(y_{real})-Var(y_{predict})}{Var(y_{real})}$

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


DGM_21, grey_model_21跟Markov_predict这仨个玩意写的太屎了，回头心情好的时候重写一遍，现在实在太困了


## 依赖
暂时先写在这里（含内建库）：  
numpy、pandas、matplotlib、seaborn、plotly、preetytable、sklearn、tensorflow、matlabengine、os、threading、multiprocessing、contextlib、typing、time、（pytorch）
非ubuntu系统需修改tools模块中的prefersettings()
