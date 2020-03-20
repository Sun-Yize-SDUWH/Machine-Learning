import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# pandas包中的DataFrame用于数据分析
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = [
    'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
]

# plt.scatter用于画散点图，在画完图后，必须加plt.show才会显示
plt.figure(1)
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
# plt.show()

# pandas中iloc函数用于选取数据，下面为选取前100行的第0、1、-1列
data = np.array(df.iloc[:100, [0, 1, -1]])
X = data[:, :-1]
y = data[:, -1]
y = np.array([1 if i == 1 else -1 for i in y])
# if判断句简写


# class类init函数，当class被调用时，init会自动运行
class Model:
    def __init__(self):
        self.w = np.ones(len(data[0]) - 1, dtype=np.float32)
        self.b = 0
        self.l_rate = 0.1
        # self.data = data
        # l_rate为步长，0 < l_rate <= 1

    def sign(self, x, w, b):
        y = np.dot(x, w) + b
        return y

    # 随机梯度下降法
    # 如果点分类错误，便使w <—— w + l_rate * y(i) * x(i)
    #                   b <—— b + l_rate * y(i)
    def fit(self, X_train, y_train):
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for d in range(len(X_train)):
                X = X_train[d]
                y = y_train[d]
                if y * self.sign(X, self.w, self.b) <= 0:
                    self.w = self.w + self.l_rate * np.dot(y, X)
                    self.b = self.b + self.l_rate * y
                    wrong_count += 1
            if wrong_count == 0:
                is_wrong = True
        return 'Perceptron Model!'

    # def score(self):
    #     pass


perceptron = Model()
print(perceptron.fit(X, y))

# linspace 4到7之间，均匀选取两个点。为4和7
x_points = np.linspace(4, 7, 2)
y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]
plt.plot(x_points, y_)
# 使用plt.plot画图是，'bo'代表散点图，如没有备注，则默认为点之间连线
plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()
