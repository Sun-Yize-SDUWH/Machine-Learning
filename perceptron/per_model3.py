"""
作业：perceptron算法

18数据科学 孙易泽
201800820135

"""
import numpy as np


class Model:
    def __init__(self, inputs):
        self.inputs = np.concatenate((inputs[:, 0:len(inputs)-1], -np.ones((len(inputs), 1))), axis=1)
        self.targets = np.array(inputs[:, len(inputs)-1])
        self.w = np.ones(len(inputs), dtype=np.float32)
        self.l_rate = 0.1
        self.time = 100

    def trainfun(self):
        m = 0
        flag = 0
        while m < self.time:
            for n in range(np.shape(self.inputs)[0]):
                outputs = np.where(np.dot(self.inputs, self.w) > 0, 1, 0)
                if outputs[n] != self.targets[n]:
                    self.w -= self.l_rate * (outputs[n] - self.targets[n]) * self.inputs[n, :]
                    outputs = np.where(np.dot(self.inputs, self.w) > 0, 1, 0)
            if all(outputs[i] == self.targets[i] for i in range(len(outputs))):
                flag = 1
                break
            else:
                m += 1
        return self.w, flag


count = 0
for i in range(16):
    i = '{:04b}'.format(i)
    x = np.array([[0, 0, int(i[0]), 0],
                  [0, 1, int(i[1]), 1],
                  [1, 0, int(i[2]), 1],
                  [1, 1, int(i[3]), 0]])
    p = Model(x)
    p.trainfun()
    m, n = p.trainfun()
    if n == 1:
        count += 1
        print('第%d组成功组合\n组合为\n%s\n权重为\n%s\n' % (count, x, m))
print('共有%d组成功的组合' % count)
