import numpy as np


class Model:
    def __init__(self):
        self.a = 1
        self.b = 2

    def sum(self):
        c = self.a + self.b
        return c


def fun1():

    print(Model.sum)


fun1()
# # np.ndim 测量数据的维度
# print(a.ndim)
# print(b.ndim)
#
# print(a.shape)
# c = np.dot(a, b)
# print(c)
