# mathon
bridge between python and matlab

## 简介
2023.11.14 <br/>
终于重写完了，从七点多写到十一点半，开会也一句话没听见<br />
以下是目前本库所有函数的简介

### connect_matlab()
本函数用于接收matlab共享的引擎接口，若不共享可以使用matlab.engine.start_matlab()来创建新的接口<br />
用法：`eng = mp.connect_matlab()`也可以传入元组

### to_matlab()
本函数主要用作本库内函数自行调用。<br />
用法：`x, y = to_matlab(x, y)`其中，x和y均为python数组

### to_matlab_gpu()
照抄的上面的函数，用法也类似只是在本库中的plot_gpu()函数中调用本函数的时候，明明传入了两个python，最后却只返回了一个matlab数组，具体是什么我没有传入matlab工作区查看，如果遇到什么不清不楚的错误需要着重注意一下<font color="red">这里</font>
<br />
用法：`x, y = to_matlab_gpu(x, y)`也可以传入元组

### to_python()
本函数
## 问题
### 一、
mp.plot_gpu()函数里面to_matlab_gpu()的调用里，给后者传入args的参数需要解包。暂时照这个做了
妈的我这个函数是直接搬的mp.plot()为什么那个函数不用解包这个函数就得解包？？？？？？？？？？？？？
离谱
args确实是元组而且确实长度为2
### 二、
利用gpu在一张图上绘制多个折线图的函数还没有写，改天在写

