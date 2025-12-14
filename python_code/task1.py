import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data[:, 2:]  # 选择花瓣长度和宽度两个特征
y = iris.target
feature_names = iris.feature_names[2:]
target_names = iris.target_names

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 定义不同的分类器
classifiers = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
    ('SVM (Linear)', SVC(kernel='linear', probability=True, random_state=42)),
    ('SVM (RBF)', SVC(kernel='rbf', probability=True, random_state=42)),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('K-Nearest Neighbors', KNeighborsClassifier(n_neighbors=5)),
    ('Gaussian Naive Bayes', GaussianNB())
]

# 创建网格用于绘制决策边界
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# 设置颜色
cmap_light = mcolors.ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = mcolors.ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# 创建子图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, (name, clf) in enumerate(classifiers):
    # 训练分类器
    clf.fit(X_train, y_train)
    
    # 在网格上预测
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    axes[i].contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    
    # 绘制训练点
    scatter = axes[i].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    
    # 计算并显示准确率
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    axes[i].set_title(f'{name} (Accuracy: {accuracy:.2f})')
    axes[i].set_xlabel(f'{feature_names[0]} (scaled)')
    axes[i].set_ylabel(f'{feature_names[1]} (scaled)')
    axes[i].set_xlim(xx.min(), xx.max())
    axes[i].set_ylim(yy.min(), yy.max())

plt.tight_layout()
plt.savefig('task1_classifier_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

