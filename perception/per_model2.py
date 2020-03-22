import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import Perceptron
print(sklearn.__version__)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = [
    'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
]

data = np.array(df.iloc[:100, [0, 1, -1]])
X = data[:, :-1]
y = data[:, -1]
y = np.array([1 if i == 1 else -1 for i in y])

clf = Perceptron(fit_intercept=True,
                 max_iter=1000,
                 shuffle=True)
clf.fit(X, y)
print(clf.coef_)
# coef_返回值为感知机中w的参数，intercept_返回为b的参数

plt.figure(figsize=(10, 10))

# 中文标题
# rcParams用于设置图像具体参数，font.sans-serif和axes用于设置图像中汉字正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('鸢尾花线性数据示例')

plt.scatter(data[:50, 0], data[:50, 1], c='b', label='Iris-setosa',)
plt.scatter(data[50:100, 0], data[50:100, 1], c='orange', label='Iris-versicolor')

# 画感知机的线
# np.arange的步长默认取1
x_ponits = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_ponits, y_)

# 其他部分
plt.legend()  # 显示图例
plt.grid(False)  # 不显示网格
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()
