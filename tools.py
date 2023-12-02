from time import time


def timer(MeasuredFunction):
    def deco(*args):
        t1 = time()
        MeasuredFunction(*args)
        t2 = time()
        print(f"{MeasuredFunction}的运行时间为{t2-t1}s")

    return deco
