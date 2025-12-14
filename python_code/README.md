# 鸢尾花分类与可视化 实验说明

本仓库的代码基于经典的鸢尾花（Iris）数据集，围绕分类算法与可视化展开。代码已按实验划分为四个独立的任务（实验），每个实验包含示例脚本用于训练模型并可视化结果。

## 四个实验（简要说明）
- **实验 1 — 二维分类比较 (`text1.py`, `classifier2d.py`)**：
	- 使用 `LogisticRegression`, `SVC`, `DecisionTreeClassifier`, `KNeighborsClassifier`, `GaussianNB` 等多种分类器。
	- 在二维特征子集上训练并绘制决策边界与样本分布，用于比较不同模型的表现。
- **实验 2 — 三维可视化与 SVM (`text2.py`)**：
	- 使用 3D 绘图（`mpl_toolkits.mplot3d`）展示三维特征空间中的样本分布与分类结果，适合探索特征间的交互关系。
- **实验 3 — 三维/色彩映射演示 (`text3.py`)**：
	- 演示使用不同的归一化与色彩映射方法来增强 3D 可视化的可读性，常用于展示概率或类别置信度。
- **实验 4 — 交互式可视化 (Plotly) (`text4.py`, `task4_interactive_3d_visualization.html`)**：
	- 使用 Plotly 构建交互式图表（可缩放、悬停信息），并生成可离线打开的 HTML 可视化结果，便于演示与分享。

此外，`data_preview.py` 用于快速预览数据分布（使用 seaborn / plotly），帮助理解数据特性。

## 环境与依赖
推荐在 Windows 下为本项目创建独立虚拟环境以隔离依赖。

1. 创建虚拟环境：
```powershell
python -m venv .venv
```

2. 激活虚拟环境（PowerShell）：
```powershell
.venv\Scripts\Activate.ps1
```

3. 升级 pip 并安装依赖：
```powershell
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -r requirements.txt
```

4. 快速验证安装：
```powershell
.venv\Scripts\python -c "import numpy,pandas,matplotlib,seaborn,sklearn,plotly; print('ok')"
```

## `requirements.txt`
项目根目录下已有 `requirements.txt`，包含本实验运行所需的主要包（例如：
`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `plotly`）。如果你需要，我可以把 `requirements.txt` 扩展为包含所有安装的依赖精确版本。

## 国内镜像（可选，建议用于加速）
在 Windows 下可通过 `%APPDATA%\pip\pip.ini` 配置 pip 镜像，例如使用阿里云镜像：
```
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
```
或切换为清华镜像：
```
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
```

