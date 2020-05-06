import random
import numpy as np
import pandas as pd


floatmin = 0.00001
random.seed(10)

class SMO:
    def __init__(self, inputs):
        self.row = len(data)
        self.col = len(data[0])-1
        self.inputs = inputs[:, :self.col]
        self.outputs = inputs[:, self.col]
        self.a = np.zeros(self.row)
        self.b = 0
        self.c = 0.05
        self.passes = 0
        self.max_passes = 1

    def mainSMO(self):
        while self.passes < self.max_passes:
            changenum = 0
            for i in range(self.row):
                self.ei = self.estfun(i)
                if (self.outputs[i] * self.ei <= 0 and self.a[i] < self.c) or (self.outputs[i] * self.ei > floatmin and self.a[i] > 0):
                    j = i
                    while j == i:
                        j = int(random.random() * (self.row - 1))
                    self.ej = self.estfun(j)
                    self.aiold = self.a[i]
                    self.ajold = self.a[j]
                    if self.outputs[i] == self.outputs[j]:
                        self.vl = max(0, self.aiold + self.ajold - self.c)
                        self.vh = min(self.c, self.aiold + self.ajold)
                    else:
                        self.vl = max(0, self.ajold - self.aiold)
                        self.vh = min(self.c, self.c + self.ajold - self.aiold)
                    if self.vh == self.vl:
                        continue
                    self.vn = self.nfun(i, j)
                    if self.vn >= 0:
                        continue
                    self.ajraw = self.ajold - self.outputs[j] * (self.ei - self.ej) / self.vn
                    if self.ajraw > self.vh:
                        self.ajnew = self.vh
                    elif self.vl <= self.ajraw <= self.vh:
                        self.ajnew = self.ajraw
                    else:
                        self.ajnew = self.vl
                    if abs(self.ajnew-self.ajold) < 0.001:
                        continue
                    self.ainew = self.aiold + self.outputs[i] * self.outputs[j] * (self.ajold - self.ajnew)
                    vb = self.bfun(i, j)
                    self.a[i] = self.ainew
                    self.a[j] = self.ajnew
                    self.b = vb
                    changenum += 1

            if changenum == 0:
                self.passes += 1
            else:
                self.passes = 0
        return self.w, self.b, self.c, self.max_passes


    def estfun(self, num):
        self.w = np.zeros(self.col)
        for i1 in range(self.row):
            self.w += self.a[i1] * self.outputs[i1] * self.inputs[i1]
        xi = np.array(self.inputs[num]).transpose()
        e = np.dot(self.w, xi) + self.b - self.outputs[num]
        return e

    def nfun(self, i, j):
        temp1 = np.dot(self.inputs[i], np.array(self.inputs[i]).transpose())
        temp2 = np.dot(self.inputs[j], np.array(self.inputs[j]).transpose())
        temp3 = 2 * np.dot(self.inputs[i], np.array(self.inputs[j]).transpose())
        n = temp3 - temp2 - temp1
        return n

    def bfun(self, i, j):
        ij = np.dot(self.inputs[i], np.array(self.inputs[j]).transpose())
        ii = np.dot(self.inputs[i], np.array(self.inputs[i]).transpose())
        jj = np.dot(self.inputs[j], np.array(self.inputs[j]).transpose())
        disj = self.ajnew - self.ajold
        disi = self.ainew - self.aiold
        b1 = self.b - self.ei - self.outputs[i] * disi * ii - self.outputs[j] * disj * ij
        b2 = self.b - self.ej - self.outputs[i] * disi * ij - self.outputs[j] * disj * jj
        if 0 < self.ainew < self.c:
            tempb = b1
        elif 0 < self.ajnew < self.c:
            tempb = b2
        else:
            tempb = (b1+b2)/2
        return tempb


def testSample(testdata, valuew, valueb):
    num = len(testdata)
    col = len(testdata[0])-1
    inputs = testdata[:, :col]
    targets = testdata[:, col]
    correct = 0
    for i in range(num):
        output = np.dot(valuew, np.array(inputs[i]).transpose())
        output += valueb
        if output * targets[i] > 0:
            correct += 1
    rate = correct / num
    return rate


dataTotal = np.array(pd.read_csv('data.csv'))
data = dataTotal[:1600, :]
testdata = dataTotal[2000:2200, :]
# print(data[0][123])

test = SMO(data)
[w, b, c, max_passes] = test.mainSMO()
rate = testSample(testdata, w, b) * 100
print("SMO Algorithm\n1600 training sets, 200 test sets\nC={0}, max_passes={1}, accuracy={2:.2f}".format(c, max_passes, rate))
