import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import plotly.io as pio
import plotly.express as px

# 设置plotly默认主题
pio.templates.default = "plotly_white"

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 选择两个类别：Setosa(0)和Versicolor(1)
mask = y != 2
X_binary = X[mask][:, 1:4]  # 萼片宽度、花瓣长度、花瓣宽度
y_binary = y[mask]
feature_names_selected = [feature_names[i] for i in [1, 2, 3]]
target_names_selected = [target_names[i] for i in [0, 1]]

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_binary)

# 训练两种分类器
lr_clf = LogisticRegression(random_state=42)
lr_clf.fit(X_scaled, y_binary)

svm_clf = SVC(kernel='linear', probability=True, random_state=42)
svm_clf.fit(X_scaled, y_binary)

# 创建网格
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
z_min, z_max = X_scaled[:, 2].min() - 0.5, X_scaled[:, 2].max() + 0.5

# 创建较粗的网格用于等值面
xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 15),
                         np.linspace(y_min, y_max, 15),
                         np.linspace(z_min, z_max, 15))

# 将网格点转换为二维数组
grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

# 预测概率
lr_probabilities = lr_clf.predict_proba(grid_points)[:, 1]  # Versicolor的概率
lr_probabilities = lr_probabilities.reshape(xx.shape)

# 预测SVM决策函数值
svm_decision = svm_clf.decision_function(grid_points)
svm_decision = svm_decision.reshape(xx.shape)

# 创建子图
fig = make_subplots(rows=1, cols=2,
                    specs=[[{'type': 'volume'}, {'type': 'volume'}]],
                    subplot_titles=('Logistic Regression Probability Map', 
                                  'SVM Decision Boundary & Confidence'))

# 添加Logistic回归概率图
fig.add_trace(go.Volume(
    x=xx.flatten(),
    y=yy.flatten(),
    z=zz.flatten(),
    value=lr_probabilities.flatten(),
    isomin=0.1,
    isomax=0.9,
    opacity=0.2,  # 整体透明度
    surface_count=15,  # 等值面数量
    colorscale='RdBu',
    reversescale=True,
    colorbar=dict(title="Probability", x=0.45, len=0.7),
    caps=dict(x_show=False, y_show=False, z_show=False),
    name='Probability'
), row=1, col=1)

# 添加SVM决策函数等值面
fig.add_trace(go.Volume(
    x=xx.flatten(),
    y=yy.flatten(),
    z=zz.flatten(),
    value=svm_decision.flatten(),
    isomin=-1.5,
    isomax=1.5,
    opacity=0.2,
    surface_count=15,
    colorscale='RdBu',
    reversescale=True,
    colorbar=dict(title="Decision Value", x=1.0, len=0.7),
    caps=dict(x_show=False, y_show=False, z_show=False),
    name='Decision Value'
), row=1, col=2)

# 添加原始数据点
colors = ['#1f77b4', '#d62728']  # 蓝色和红色
markers = ['circle', 'diamond']

for i, target in enumerate(np.unique(y_binary)):
    mask = (y_binary == target)
    # 添加到Logistic回归图
    fig.add_trace(go.Scatter3d(
        x=X_scaled[mask, 0],
        y=X_scaled[mask, 1],
        z=X_scaled[mask, 2],
        mode='markers',
        marker=dict(size=8, color=colors[i], symbol=markers[i],
                   line=dict(width=2, color='white')),
        name=f'{target_names_selected[i]} (LR)',
        showlegend=(i == 0)
    ), row=1, col=1)
    
    # 添加到SVM图
    fig.add_trace(go.Scatter3d(
        x=X_scaled[mask, 0],
        y=X_scaled[mask, 1],
        z=X_scaled[mask, 2],
        mode='markers',
        marker=dict(size=8, color=colors[i], symbol=markers[i],
                   line=dict(width=2, color='white')),
        name=f'{target_names_selected[i]} (SVM)',
        showlegend=(i == 0)
    ), row=1, col=2)

# 添加0概率等值面（决策边界）到SVM图
fig.add_trace(go.Isosurface(
    x=xx.flatten(),
    y=yy.flatten(),
    z=zz.flatten(),
    value=svm_decision.flatten(),
    isomin=0,
    isomax=0,
    colorscale=[[0, 'black'], [1, 'black']],
    surface_count=1,
    opacity=0.6,
    showscale=False,
    name='Decision Boundary',
    showlegend=True
), row=1, col=2)

# 设置布局
fig.update_layout(
    title_text="3D Classification Visualization: Probability Maps & Decision Boundaries",
    title_x=0.5,
    width=1400,
    height=700,
    scene=dict(
        xaxis_title=f'{feature_names_selected[0]} (scaled)',
        yaxis_title=f'{feature_names_selected[1]} (scaled)',
        zaxis_title=f'{feature_names_selected[2]} (scaled)',
        aspectmode='cube',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    ),
    scene2=dict(
        xaxis_title=f'{feature_names_selected[0]} (scaled)',
        yaxis_title=f'{feature_names_selected[1]} (scaled)',
        zaxis_title=f'{feature_names_selected[2]} (scaled)',
        aspectmode='cube',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# 保存为HTML文件
fig.write_html("task4_interactive_3d_visualization.html")

# 显示图形
fig.show()

