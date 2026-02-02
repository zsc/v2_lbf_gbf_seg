# Coupled LBF+GBF 水平集图像分割演示

本项目实现了一个基于 **LBF（Local Binary Fitting，局部二值拟合）** 和 **GBF（Global Binary Fitting，全局二值拟合）** 的**耦合水平集（Coupled Level Set）**图像分割算法。支持 FastAPI 后端服务和命令行测试工具。

---

## 📖 算法原理

### 1. 水平集方法（Level Set Method）

水平集方法通过隐式函数 $\phi(x, y)$ 来表示演化曲线 $C$，其中：

$$
C = \{(x, y) \mid \phi(x, y) = 0\}
$$

- 当 $\phi(x, y) > 0$：点在曲线内部
- 当 $\phi(x, y) < 0$：点在曲线外部
- 当 $\phi(x, y) = 0$：点在曲线上

### 2. Heaviside 函数和 Dirac 函数

为了处理水平集函数，我们使用正则化的 Heaviside 函数和 Dirac 函数：

**Heaviside 函数**（平滑阶跃函数）：

$$
H_\varepsilon(\phi) = \frac{1}{2} \left[ 1 + \frac{2}{\pi} \arctan\left(\frac{\phi}{\varepsilon}\right) \right]
$$

**Dirac 函数**（平滑脉冲函数）：

$$
\delta_\varepsilon(\phi) = \frac{1}{\pi} \cdot \frac{\varepsilon}{\varepsilon^2 + \phi^2}
$$

其中 $\varepsilon$ 是控制平滑程度的参数。

### 3. LBF（局部二值拟合）模型

LBF 模型利用局部灰度信息，适合处理灰度不均匀的图像。定义局部拟合能量：

$$
\mathcal{E}_{LBF}(C, f_1, f_2) = \lambda_1 \int \left( \int_{inside(C)} K_\sigma(x-y) |I(y) - f_1(x)|^2 dy \right) dx + \lambda_2 \int \left( \int_{outside(C)} K_\sigma(x-y) |I(y) - f_2(x)|^2 dy \right) dx
$$

其中：
- $K_\sigma$ 是高斯核函数，用于提取局部信息
- $f_1(x)$ 和 $f_2(x)$ 是空间变化的局部拟合函数

在本实现中，使用高斯卷积近似：

```python
f1 = conv(I * H, k) / conv(H, k)
f2 = conv(I * (1-H), k) / conv(1-H, k)
```

### 4. GBF（全局二值拟合）模型

GBF 模型利用全局灰度统计信息，类似于 Chan-Vese 模型：

$$
\mathcal{E}_{GBF}(C, c_1, c_2) = \lambda_1 \int_{inside(C)} |I(x) - c_1|^2 dx + \lambda_2 \int_{outside(C)} |I(x) - c_2|^2 dx
$$

其中 $c_1$ 和 $c_2$ 分别是曲线内外的平均灰度值：

$$
c_1 = \frac{\int_\Omega I(x) \cdot H(\phi) dx}{\int_\Omega H(\phi) dx}, \quad c_2 = \frac{\int_\Omega I(x) \cdot (1 - H(\phi)) dx}{\int_\Omega (1 - H(\phi)) dx}
$$

### 5. 耦合模型（Coupled Model）

本项目的核心是将 LBF 和 GBF 结合，充分利用局部和全局信息：

$$
\mathcal{E}_{total} = \omega \cdot \mathcal{E}_{GBF} + (1-\omega) \cdot \mathcal{E}_{LBF} + \mathcal{E}_{length} + \mathcal{E}_{distance}
$$

或在本实现中采用的可调权重形式：

$$
\mathcal{E}_{total} = \mathcal{E}_{LBF} + \omega \cdot \mathcal{E}_{GBF} + \nu \cdot \mathcal{E}_{length} + \mu \cdot \mathcal{E}_{distance}
$$

**各项说明：**

| 能量项 | 公式 | 作用 |
|--------|------|------|
| 数据项 (LBF) | $\int (e_1 \cdot H + e_2 \cdot (1-H))$ | 局部灰度拟合 |
| 数据项 (GBF) | $\omega \cdot \int [(I-c_1)^2 \cdot H + (I-c_2)^2 \cdot (1-H)]$ | 全局灰度拟合 |
| 长度项 | $\nu \int \delta(\phi) |\nabla \phi| dx$ | 平滑曲线，去噪 |
| 距离惩罚项 | $\mu \int \frac{1}{2}(|\nabla \phi| - 1)^2 dx$ | 保持 $|\nabla \phi| \approx 1$ |

### 6. 演化方程

水平集函数的演化遵循梯度下降：

$$
\frac{\partial \phi}{\partial t} = -\delta_\varepsilon(\phi) \cdot F_{data} + \nu \cdot \delta_\varepsilon(\phi) \cdot \kappa + \mu \cdot \left[ \nabla^2 \phi - \text{div}\left(\frac{\nabla \phi}{|\nabla \phi|}\right) \right]
$$

其中：
- $F_{data}$：数据驱动力（来自 LBF 和/或 GBF）
- $\kappa = \text{div}\left(\frac{\nabla \phi}{|\nabla \phi|}\right)$：曲率
- 最后一项是距离保持正则化

---

## 🗂️ 代码结构

```
.
├── app.py              # FastAPI 后端服务
├── cli_test.py         # 命令行测试工具
├── seg/                # 核心算法包
│   ├── levelset.py     # 水平集演化和能量计算
│   ├── kernels.py      # 高斯核生成
│   └── viz.py          # 可视化工具
├── static/             # 前端静态文件
│   ├── index.html      # Web 界面
│   ├── app.js          # 前端逻辑
│   └── style.css       # 样式表
├── requirements.txt    # Python 依赖
└── test.png           # 测试图像
```

---

## 🔧 核心代码说明

### 1. 高斯核生成 (`seg/kernels.py`)

```python
def gaussian_kernel2d(sigma: float, ksize: int | None = None, ...) -> torch.Tensor:
```

生成二维高斯核，用于局部信息提取。核大小自动计算为 $2 \times \lceil 3\sigma \rceil + 1$。

### 2. 水平集参数 (`seg/levelset.py`)

```python
@dataclass(frozen=True)
class LevelSetParams:
    model: str = "coupled"      # "lbf" | "gbf" | "coupled"
    sigma: float = 3.0          # 高斯核标准差
    epsilon: float = 1.5        # Heaviside/Dirac 平滑参数
    nu: float = 0.3             # 长度项权重
    mu: float = 0.3             # 距离惩罚项权重
    omega: float = 1.0          # GBF 项权重（耦合模式）
    dt: float = 0.1             # 时间步长
    iters: int = 120            # 迭代次数
    viz_stride: int = 5         # 可视化采样间隔
    init_type: str = "otsu"     # 初始化方式
```

### 3. 核心演化循环 (`seg/levelset.py`)

主要迭代流程：

```python
for it in range(0, params.iters + 1):
    # 1. 计算 Heaviside 和 Dirac 函数
    H = _heaviside(phi, params.epsilon)
    delta = _dirac(phi, params.epsilon)
    
    # 2. LBF 计算：局部拟合函数 f1, f2 和能量 e1, e2
    if want_lbf:
        f1 = conv(I*H, k) / conv(H, k)
        f2 = conv(I*(1-H), k) / conv(1-H, k)
        e1 = I^2 - 2*I*conv(f1,k) + conv(f1^2,k)
        e2 = I^2 - 2*I*conv(f2,k) + conv(f2^2,k)
    
    # 3. GBF 计算：全局平均值 c1, c2
    if want_gbf:
        c1 = sum(I*H) / sum(H)
        c2 = sum(I*(1-H)) / sum(1-H)
        gbf_term = (I-c1)^2 - (I-c2)^2
    
    # 4. 组合数据驱动力
    data_force = (e1 - e2) + omega * gbf_term
    
    # 5. 计算曲率和距离惩罚
    kappa = divergence(nx, ny)  # 曲率
    rphi = divergence(rx, ry)   # 距离惩罚
    
    # 6. 更新水平集函数
    rhs = -delta*data_force + nu*delta*kappa + mu*rphi
    phi = phi + dt * rhs
```

### 4. 初始化方法

支持三种初始化方式：

- **Otsu 阈值** (`otsu`): 自动计算最优阈值进行二值化
  ```python
  thr = argmax[(μ_T·ω - μ)² / (ω(1-ω))]
  ```
  
- **圆形** (`circle`): 以图像中心为圆心，半径为图像短边的比例
  
- **矩形** (`rect`): 以图像中心为矩形，边距为图像尺寸的比例

---

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动 Web 服务

```bash
python app.py
```

然后打开浏览器访问 `http://127.0.0.1:8000/`

### 命令行测试

```bash
# 使用默认参数运行
python cli_test.py

# 指定模型和参数
python cli_test.py --model coupled --sigma 3.0 --iters 120 --omega 1.0

# 使用圆形初始化
python cli_test.py --init-type circle --radius-frac 0.25
```

---

## ⚙️ 参数说明

| 参数 | 说明 | 默认值 | 范围 |
|------|------|--------|------|
| `model` | 模型类型: lbf/gbf/coupled | `coupled` | - |
| `sigma` | 高斯核标准差，控制局部区域大小 | `3.0` | > 0 |
| `epsilon` | Heaviside/Dirac 平滑参数 | `1.5` | > 0 |
| `nu` | 曲线长度惩罚权重 | `0.3` | ≥ 0 |
| `mu` | 距离惩罚权重，保持符号距离函数特性 | `0.3` | ≥ 0 |
| `omega` | GBF 权重（仅耦合模式） | `1.0` | ≥ 0 |
| `dt` | 时间步长 | `0.1` | > 0 |
| `iters` | 迭代次数 | `120` | ≥ 1 |

---

## 🖥️ 硬件加速

在 Apple Silicon (M1/M2/M3) 设备上，自动使用 MPS（Metal Performance Shaders）加速：

```python
device = torch.device("mps")  # 若可用
```

---

## 📊 输出结果

Web 界面和 CLI 工具会返回以下信息：

- **overlay**: 原图叠加分割轮廓
- **mask**: 二值掩码
- **f1/f2**: 局部拟合函数（LBF 模式）
- **energy**: 总能量
- **energy_lbf_data**: LBF 数据项能量
- **energy_gbf_data**: GBF 数据项能量
- **energy_len**: 长度项能量
- **energy_dist**: 距离惩罚项能量
