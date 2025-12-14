import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# 选择两个类别：Setosa(0)和Versicolor(1)
mask = y != 2
X_binary = X[mask][:, 1:4]  # 选择萼片宽度、花瓣长度、花瓣宽度
y_binary = y[mask]
selected_features = [feature_names[i] for i in [1, 2, 3]]

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_binary)

# 训练逻辑回归分类器
clf = LogisticRegression(random_state=42)
clf.fit(X_scaled, y_binary)

# 创建网格
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
z_min, z_max = X_scaled[:, 2].min() - 0.5, X_scaled[:, 2].max() + 0.5

# 生成网格点（减少分辨率以提高性能）
xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 15),
                         np.linspace(y_min, y_max, 15),
                         np.linspace(z_min, z_max, 15))

# 将网格点转换为二维数组
grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

# 预测每个网格点属于类别1（Versicolor）的概率
probabilities = clf.predict_proba(grid_points)[:, 1]  # Versicolor类的概率
probabilities = probabilities.reshape(xx.shape)

# 创建3D图形
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# 创建自定义颜色映射，增强可视化效果
colors = [(0.2, 0.4, 1.0), (0.6, 0.8, 1.0), (1.0, 1.0, 1.0), (1.0, 0.8, 0.6), (1.0, 0.4, 0.2)]  # 蓝-白-红
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom_blue_white_red', colors, N=n_bins)

# 为不同概率阈值创建多个半透明等值面
thresholds = [0.2, 0.4, 0.5, 0.6, 0.8]
alphas = [0.1, 0.2, 0.3, 0.2, 0.1]
colors = ['blue', 'lightblue', 'white', 'lightcoral', 'red']

for i, threshold in enumerate(thresholds):
    # 寻找接近阈值的点
    mask = np.abs(probabilities - threshold) < 0.03
    
    if np.any(mask):
        # 提取这些点的坐标
        x_surf = xx[mask]
        y_surf = yy[mask]
        z_surf = zz[mask]
        c_surf = np.ones_like(x_surf) * threshold
        
        # 绘制等值面
        ax.scatter(x_surf, y_surf, z_surf, c=c_surf, cmap=cmap, 
                  vmin=0, vmax=1, alpha=alphas[i], s=15, edgecolors='none')

# 绘制原始数据点
setosa_points = ax.scatter(X_scaled[y_binary == 0, 0], 
                          X_scaled[y_binary == 0, 1], 
                          X_scaled[y_binary == 0, 2],
                          c='darkblue', marker='o', s=80, label='Setosa (Actual)', edgecolors='k', linewidth=1)

versicolor_points = ax.scatter(X_scaled[y_binary == 1, 0], 
                              X_scaled[y_binary == 1, 1], 
                              X_scaled[y_binary == 1, 2],
                              c='darkred', marker='^', s=80, label='Versicolor (Actual)', edgecolors='k', linewidth=1)

# 添加颜色条
sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=8, pad=0.1)
cbar.set_label('Probability of Versicolor', fontsize=12)
cbar.set_ticks(np.linspace(0, 1, 6))

# 设置标签和标题
ax.set_xlabel(f'\n{selected_features[0]} (scaled)', linespacing=3.2, fontsize=12)
ax.set_ylabel(f'\n{selected_features[1]} (scaled)', linespacing=3.2, fontsize=12)
ax.set_zlabel(f'\n{selected_features[2]} (scaled)', linespacing=3.2, fontsize=12)
ax.set_title('3D Probability Map for Logistic Regression\nSetosa vs Versicolor', fontsize=14, pad=20)
ax.legend(loc='upper left')

# 调整视角
ax.view_init(elev=25, azim=40)

# 设置背景色
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('task3_3d_probability_map.png', dpi=300, bbox_inches='tight')
plt.show()

