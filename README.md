# mathon
bridge between python and matlab

# 问题
## 一、
mp.plot_gpu()函数里面to_matlab_gpu()的调用里，给后者传入args的参数需要解包。暂时照这个做了
妈的我这个函数是直接搬的mp.plot()为什么那个函数不用解包这个函数就得解包？？？？？？？？？？？？？
离谱
args确实是元组而且确实长度为2
## 二、
利用gpu在一张图上绘制多个折线图的函数还没有写，改天在写

