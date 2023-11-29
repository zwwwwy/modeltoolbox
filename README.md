# mathon
bridge between python and matlab

## 简介
2023.11.14 <br/>
终于重写完了，从七点多写到十一点半，开会也一句话没听见<br />
以下是目前本库所有函数的简介

### `connect_matlab()`
本函数用于接收matlab共享的引擎接口，若不共享可以使用matlab.engine.start_matlab()来创建新的接口<br />
用法：`eng = mp.connect_matlab()`也可以传入元组<br/>
这个函数的行为被改成连接最后一个共享的matlab接口了，因为linux的matlab做的实在是一坨，关掉一个matlab以后他共享的接口居然是关不掉的，而且这个破matlab还老是把opengl的设置改回去，搞得我好几次都要开两遍接口，真的烦

### `to_matlab(*args)`
本函数主要用作本库内函数自行调用。<br />
用法：`x, y = to_matlab(x, y)`其中，x和y均为python数组

### `to_matlab_gpu(eng, *args)`
照抄的上面的函数，用法也类似只是在本库中的plot_gpu()函数中调用本函数的时候，明明传入了两个python，最后却只返回了一个matlab数组，具体是什么我没有传入matlab工作区查看，如果遇到什么不清不楚的错误需要着重注意一下<font color="red">这里</font>
<br />
用法：`x, y = to_matlab_gpu(eng, x, y)`也可以传入元组

### `to_python(*args)`
本函数把MATLAB数组转化为python数组（不含gpuArray）<br />
用法：`x, y = to_python(x, y)`

### `matlab_gpu_to_python(eng, *args)`
本函数把gpuArray转化为python数组<br/>
用法：`x, y = matlab_gpu_to_python(eng, x, y)`

### `to_workspace(eng, dic)`
本函数把在python中储存的变量传入matlab工作区，仅支持传入matlab数据（未测试过python数组能否自动处理），可能需要结合前面的函数使用。<br/>
注意：本函数是以字典的形式传入工作区，键是变量名，值是变量值。例如如果想把1传入matlab的工作区，并赋予其a的名称，则需要传入{'a':1}。这样在matlab的工作区中则会出现该a=1的变量。<br/>
用法：`to_workspace(eng, {'a':1})`

### `read_workspace(eng, *args)`
本函数用于读取matlab中工作区的变量传入python，使用时输入字符型的matlab变量名。目前本函数用的不多，也没怎么测试，如果输入matlab中不存在的变量名可能会出错。本函数返回一个字典，字典的意义参见上个函数。<br/>
用法：`read_workspace(eng, 'a', 'b')`

### `add_workspace(eng, x, y, z=None)`
这个函数我记得就是为了给下面几个绘图的函数用而临时写的，就是为了在画图的时候可以把自变量和因变量传入matlab，所以用法不写了。

### `plot(eng, *args)`
以x、y、z的顺序传入两个或者三个变量，利用matlab绘制一个二维曲线或者三维曲线图。<br/>
用法：`plot(eng, x, y)`

### `plot_gpu(eng, *args)`
plot函数的gpu版本，这个函数有点离谱，明明跟上面的函数结构一样，转换数组类型的时候这个函数就要解包一下，上面内个就不用，原因不明，可能是因为我在一行里面把数据类型换了两次？用的时候可能会出错。<br/>
后面考虑把画图函数分成本体和装饰器什么的重写一遍。<br/>
用法：`plot_gpu(eng, x, y)`

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
用法：`mp.mult_plot_str(eng, [[1,2,3], [2,3,6]], ['a','b'])`

### `mult_plot_str_gpu()`
gpu版本，用法一样

### `preview(func)`
装饰器，用来运行plt.show()

### `save_plt_fig(path, hold)`
保存图片的函数，默认放在当前文件目录下的pypic文件夹中，可以手动选定路径，保存的图片依照先后顺序按数字从小到大命名。默认画完当前图像后自动hold off，如果需要hold on，给hold传入'on'就可以了。<br/>
用法：`save_plt_fig(path='./pic', hold='off')`

### SavePreview_plt_fig():
跟上面的函数用法完全相同，保存以后会自动跳出函数图像的图像，我觉得这里没办法设置hold on，但是还是写了一个判断传入的hold是on还是off的功能。

## 问题
mp.plot_gpu()函数里面to_matlab_gpu()的调用里，给后者传入args的参数需要解包。暂时照这个做了
妈的我这个函数是直接搬的mp.plot()为什么那个函数不用解包这个函数就得解包？？？？？？？？？？？？？
离谱
args确实是元组而且确实长度为2

