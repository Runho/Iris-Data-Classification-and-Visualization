import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import LightSource

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# 选择两个类别：Setosa(0)和Versicolor(1)，排除Virginica(2)
mask = y != 2
X_binary = X[mask][:, 1:4]  # 选择萼片宽度、花瓣长度、花瓣宽度三个特征
y_binary = y[mask]
selected_features = [feature_names[i] for i in [1, 2, 3]]

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_binary)

# 训练SVM分类器
clf = SVC(kernel='linear', random_state=42)
clf.fit(X_scaled, y_binary)

# 创建网格用于绘制3D决策边界
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5

# 创建网格点
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20),
                     np.linspace(y_min, y_max, 20))

# 计算决策边界平面
w = clf.coef_[0]
b = clf.intercept_[0]
zz = (-w[0] * xx - w[1] * yy - b) / w[2]

# 创建3D图形
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 设置光源增强3D效果
ls = LightSource(azdeg=0, altdeg=65)
rgb = ls.shade(zz, plt.cm.viridis)

# 绘制决策边界平面
surf = ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, facecolors=rgb,
                      linewidth=0, antialiased=True, alpha=0.7)

# 绘制数据点
setosa_points = ax.scatter(X_scaled[y_binary == 0, 0], 
                          X_scaled[y_binary == 0, 1], 
                          X_scaled[y_binary == 0, 2],
                          c='blue', marker='o', s=60, label='Setosa', edgecolors='k')

versicolor_points = ax.scatter(X_scaled[y_binary == 1, 0], 
                              X_scaled[y_binary == 1, 1], 
                              X_scaled[y_binary == 1, 2],
                              c='red', marker='^', s=60, label='Versicolor', edgecolors='k')

# 添加网格线，增强3D效果
ax.grid(True, linestyle='--', alpha=0.7)

# 设置标签和标题
ax.set_xlabel(f'\n{selected_features[0]} (scaled)', linespacing=3.2)
ax.set_ylabel(f'\n{selected_features[1]} (scaled)', linespacing=3.2)
ax.set_zlabel(f'\n{selected_features[2]} (scaled)', linespacing=3.2)
ax.set_title('3D Decision Boundary for SVM Classifier\nSetosa vs Versicolor', fontsize=14, pad=20)
ax.legend(loc='upper left')

# 调整视角
ax.view_init(elev=20, azim=35)

plt.tight_layout()
plt.savefig('task2_3d_decision_boundary.png', dpi=300, bbox_inches='tight')
plt.show()

