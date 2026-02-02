下面我把你给出的推导“从符号统一→卷积引理的本质→(f_1,f_2) 与 (c_1,c_2) 的极小化→对 (\phi) 的欧拉–拉格朗日方程→数值实现要点”的逻辑完整细说一遍；最后给出一个**可直接交给 codex / gemini-cli 生成代码**的 **Python + HTML 交互 Demo SPEC（中文 Markdown，含数学公式）**。

---

## 1) 先把记号钉死：谁是像素坐标、谁是局部拟合中心

你的式(1)里出现了双重积分 (\int d\mathbf x \int d\mathbf y)，而 LBF 这类模型的关键是：

* **(\mathbf x)** 往往表示“当前像素位置（要被分到 inside/outside 的位置）”；
* **(\mathbf y)** 往往表示“卷积核邻域内的采样位置 / 局部统计所覆盖的点”。

但在不同文章/实现里，(\mathbf x,\mathbf y) 角色经常互换（尤其当核 (K) 对称时），最终只要你在：

* **卷积定义**
* **Heaviside (H(\phi)) 作用在哪个变量上**
* **对 (f_1,f_2) 求变分时把谁当作自变量**

这三处保持一致，就不会出错。

为了对齐你后续得到的结论
[
f_1(\mathbf x)=\frac{K_\sigma * (I,H(\phi))}{K_\sigma * H(\phi)},\qquad
f_2(\mathbf x)=\frac{K_\sigma * (I,(1-H(\phi)))}{K_\sigma * (1-H(\phi))}
]
一种最“干净”的写法是把 LBF 的 inside 项写成（与许多经典推导等价）：

[
E_{\text{in}}^{\text{LBF}}
==========================

\int_{\Omega} H(\phi(\mathbf x)),
\Big(
\int_{\Omega} K_\sigma(\mathbf y-\mathbf x),\lvert I(\mathbf x)-f_1(\mathbf y)\rvert^2,d\mathbf y
\Big),d\mathbf x
]
outside 项同理，把 (H) 换成 (1-H)、把 (f_1) 换成 (f_2)。

**直觉**：

* (H(\phi(\mathbf x))) 决定像素 (\mathbf x) 是 inside 还是 outside；
* 内层积分用核 (K_\sigma) 把邻域里“拟合函数 (f_1(\mathbf y))”的影响聚合过来，于是 (f_1) 可以“随位置变化”，用来吃亮度不均匀。

---

## 2) 你引入的卷积引理(3)本质是什么：卷积算子的“自伴随/转置”关系

你写的引理（1D）：
[
\int (f*g),h,dx = \int (f*h),g,dx
]
严格来说，它成立的充分条件是：

* 卷积按 ((f*g)(x)=\int f(x-\tau)g(\tau),d\tau) 定义；
* 并且 (f) 是中心对称 (f(x)=f(-x))，或者更一般地写成“翻转核”：

更通用的版本（也更适合 2D/ND）是：
[
\int (f*g)(x),h(x),dx
=====================

\int g(x),(\tilde f*h)(x),dx,\quad \tilde f(x)=f(-x)
]
当 (f) 中心对称时 (\tilde f=f)，就退化为你写的(3)。

**你用它做的事**：把“卷积落在 (f_1) 上”的形式，换成“卷积落在 (I\cdot H) 上”的形式——这样对 (f_1) 求变分就变成局部的点态二次函数求极值。

---

## 3) 为什么 LBF 的 (f_1,f_2) 能闭式解：局部加权最小二乘的逐点解

你已经给出了关键推导思路，我这里用“逐点最小二乘”的角度把它说清楚（对应你式(5)(6)的结论）。

### 3.1 inside 项对 (f_1) 的依赖是“每个位置一元二次函数”

把 inside 项写成：
[
J(f_1,\phi)=\int H(\phi(\mathbf x))\int K_\sigma(\mathbf y-\mathbf x),\lvert I(\mathbf x)-f_1(\mathbf y)\rvert^2,d\mathbf y,d\mathbf x
]

交换积分次序（用 Fubini + 你的引理本质），得到它可以写成对每个 (\mathbf y) 的加权最小二乘：

[
J(f_1,\phi)
===========

\int
\Big[
f_1(\mathbf y)^2, (K_\sigma * H)(\mathbf y)
-------------------------------------------

2 f_1(\mathbf y), (K_\sigma * (I H))(\mathbf y)
+
(K_\sigma * (I^2 H))(\mathbf y)
\Big],d\mathbf y
]

注意最后一项 ((K_\sigma*(I^2H))) **与 (f_1) 无关**，所以对 (f_1) 最小化只看前两项。

于是逐点求导（变分导数）：
[
\frac{\delta J}{\delta f_1}(\mathbf y)
======================================

## 2 f_1(\mathbf y),(K_\sigma * H)(\mathbf y)

2 (K_\sigma*(I H))(\mathbf y)
]
令其为 0，得到闭式解：
[
f_1(\mathbf y)=\frac{(K_\sigma*(I H))(\mathbf y)}{(K_\sigma*H)(\mathbf y)}
]

### 3.2 (f_2) 同理（把 (H) 换成 (1-H)）

[
f_2(\mathbf y)=\frac{(K_\sigma*(I (1-H)))(\mathbf y)}{(K_\sigma*(1-H))(\mathbf y)}
]

这与你给出的(6)一致。

> 实现里一定要给分母加一个很小的 (\varepsilon) 防止除零（当某处 (H) 全为 0 或全为 1 时会发生）。

---

## 4) GBF 的 (c_1,c_2) 为什么更简单：全局均值的标量最小二乘

你的式(2)就是经典“全局两相均值拟合”（很像 Chan–Vese 的数据项部分）：

[
E^{GBF}=
\int (I-c_1)^2 H(\phi),d\mathbf x
+
\int (I-c_2)^2 (1-H(\phi)),d\mathbf x
]

对标量 (c_1) 求偏导并令 0：
[
\frac{\partial E}{\partial c_1}=2\int (c_1-I)H,d\mathbf x=0
\Rightarrow
c_1=\frac{\int I H}{\int H}
]
(c_2) 同理。

---

## 5) 对 (\phi) 求变分导数：核心就是 (\frac{dH}{d\phi}=\delta)

### 5.1 平滑 Heaviside 与 Dirac（实现必须做）

数值上通常用平滑版本：

[
H_\epsilon(z)=\frac12\Big(1+\frac{2}{\pi}\arctan\frac{z}{\epsilon}\Big),
\qquad
\delta_\epsilon(z)=H_\epsilon'(z)=\frac{1}{\pi}\frac{\epsilon}{\epsilon^2+z^2}
]

其中 (\epsilon) 控制“边界带宽”。

---

### 5.2 LBF 数据项对 (\phi) 的导数长什么样

把 LBF 数据项写成：
[
E_{\text{data}}^{\text{LBF}}
============================

\int
\Big(
e_1(\mathbf x),H(\phi(\mathbf x))
+
e_2(\mathbf x),(1-H(\phi(\mathbf x)))
\Big),d\mathbf x
]
这里
[
e_1(\mathbf x)=\int K_\sigma(\mathbf y-\mathbf x),\lvert I(\mathbf x)-f_1(\mathbf y)\rvert^2,d\mathbf y,
\quad
e_2(\mathbf x)=\int K_\sigma(\mathbf y-\mathbf x),\lvert I(\mathbf x)-f_2(\mathbf y)\rvert^2,d\mathbf y
]

对 (\phi) 变分时，把 (f_1,f_2) 视作已用当前 (\phi) 更新过的“常量场”（典型做法是交替最小化），那么：

[
\frac{\delta E_{\text{data}}^{\text{LBF}}}{\delta \phi}(\mathbf x)
==================================================================

\delta(\phi(\mathbf x)),(e_1(\mathbf x)-e_2(\mathbf x))
]

因此梯度下降会给出推动项：
[
-\delta(\phi),(e_1-e_2)
]
直觉：如果某点更像 outside（(e_2<e_1)），就推动 (\phi) 往 outside 方向改符号。

**实现上的高效计算**：
展开 (e_1) 你会发现只需要卷积 (f_1)、(f_1^2)（因为 (I(\mathbf x)) 对 (\mathbf y) 是常数）：
[
e_1
===

I^2
-2 I,(K_\sigma*f_1)
+
(K_\sigma*f_1^2)
]
前提是 (K) 归一化（(\int K = 1) 或离散求和为 1）；(e_2) 同理。

---

### 5.3 GBF 数据项对 (\phi) 的导数

[
E^{GBF}=
\int (I-c_1)^2 H(\phi),d\mathbf x
+
\int (I-c_2)^2 (1-H(\phi)),d\mathbf x
]
因此
[
\frac{\delta E^{GBF}}{\delta \phi}
==================================

\delta(\phi)\Big((I-c_1)^2-(I-c_2)^2\Big)
]

耦合权重 (\omega) 加上即可。

---

### 5.4 边界长度正则项（曲率项）

你写的：
[
\nu\int |\nabla H(\phi)|,d\mathbf x
]
利用 (\nabla H(\phi)=\delta(\phi)\nabla\phi)，得：
[
\int |\nabla H(\phi)|=\int \delta(\phi),|\nabla\phi|
]
它的欧拉–拉格朗日项（经典结果）对应平均曲率流：
[
\frac{\delta}{\delta\phi}\Big(\nu\int \delta(\phi),|\nabla\phi|\Big)
====================================================================

-\nu,\delta(\phi),\mathrm{div}\Big(\frac{\nabla\phi}{|\nabla\phi|}\Big)
]
所以梯度下降里出现：
[
+\nu,\delta(\phi),\mathrm{div}\Big(\frac{\nabla\phi}{|\nabla\phi|}\Big)
]
其中
(\kappa=\mathrm{div}(\nabla\phi/|\nabla\phi|)) 就是曲率。

---

### 5.5 距离正则项（DRLSE 风格，避免频繁重初始化）

你写的：
[
\frac{\mu}{2}\int (|\nabla\phi|-1)^2,d\mathbf x
]
其变分导数可以写成：
[
\frac{\delta}{\delta\phi}
=========================

-\mu,\mathrm{div}\Big(\frac{|\nabla\phi|-1}{|\nabla\phi|}\nabla\phi\Big)
]
所以梯度下降里是：
[
+\mu,\mathrm{div}\Big(\frac{|\nabla\phi|-1}{|\nabla\phi|+\eta}\nabla\phi\Big)
]
实现时加 (\eta) 防止除零。

---

## 6) 把所有项合在一起：最终的 (\phi) 演化方程

你耦合能量：
[
E(\phi)=E^{LBF}(f_1,f_2,\phi)+\omega E^{GBF}(c_1,c_2,\phi)
]

采用梯度下降（时间步进）：
[
\frac{\partial \phi}{\partial t}=-\frac{\delta E}{\delta \phi}
]

把上面各项拼起来，一种常见写法是：

[
\boxed{
\frac{\partial \phi}{\partial t}
================================

-\delta_\epsilon(\phi)\Big[(e_1-e_2)+\omega\big((I-c_1)^2-(I-c_2)^2\big)\Big]
+\nu,\delta_\epsilon(\phi),\kappa
+\mu,\mathrm{div}\Big(\frac{|\nabla\phi|-1}{|\nabla\phi|+\eta}\nabla\phi\Big)
}
]

离散实现就是：
[
\phi^{(k+1)}=\phi^{(k)}+\Delta t\cdot RHS(\phi^{(k)})
]

---

## 7) 关于用自动求导做这类问题：一个非常关键的“detach 选择”

你提到用 PyTorch 自动求导来做，这在教学/验证推导时很爽，但要注意一个“会改变算法含义”的点：

* 经典推导通常是**交替最小化**：

  1. 用当前 (\phi) 计算 (f_1,f_2,c_1,c_2)（闭式解）；
  2. 固定它们，对 (\phi) 做一步 PDE/梯度下降；
  3. 重复。

* 如果你在 autograd 里把 (f_1,f_2,c_1,c_2) 都写成 (\phi) 的可微函数且不截断梯度，优化器实际在做的是**“把闭式解也当作计算图的一部分”**的端到端梯度，这与交替最小化并不等价，数值行为可能更敏感、更难调。

因此 Demo 里建议提供两个模式：

* **ALT（交替最小化）模式**：更新 (\phi) 时对 (f_1,f_2,c_1,c_2) 进行 `detach`（或实现显式 PDE）。更贴近论文/传统算法。
* **E2E（端到端）模式**：不 detach，纯 autograd。更“现代”，但更不稳定。

---

# Python + HTML 交互 Demo SPEC（中文 Markdown，含公式）

> 目标：给 codex / gemini-cli 一份“足够具体、可直接生成工程代码”的规格说明。
> 不限定你用哪个生成器；这里把交互、接口、算法细节、边界条件、验收标准都写清楚。

---

## 0. 项目名称

**Coupled LBF+GBF Level Set Segmentation Demo（耦合局部/全局拟合的水平集分割演示）**

---

## 1. Demo 目标与非目标

### 1.1 目标（Goals）

1. 在浏览器里上传一张图片（灰度为主），运行 LBF / GBF / Coupled 三种分割，并可视化：

   * 当前迭代的轮廓（(\phi=0) 等值线）叠加在原图上
   * 二值 mask（inside/outside）
   * 关键中间量：(f_1,f_2)（局部拟合图）、(c_1,c_2)（全局均值数值）
   * 能量曲线 (E(\phi)) 随迭代变化（可选但强烈建议）
2. 支持调整主要超参数并重新运行：

   * (\sigma)（高斯核尺度）、核大小
   * (\epsilon)（平滑 Heaviside/Dirac）
   * (\nu,\mu,\omega)（长度正则、距离正则、全局耦合权重）
   * (\Delta t)、迭代次数、每隔多少步输出一帧
3. 支持初始化 (\phi) 的至少两种方式：

   * 居中圆（半径可调）
   * 矩形框（边距可调）
   * （加分）用户在 Canvas 上手绘初始 mask，后台转换为 SDF 作为 (\phi)

### 1.2 非目标（Non-Goals）

* 不追求对复杂自然图的泛化能力（这不是深度学习模型）。
* 不做 GPU 加速要求；CPU 可跑即可。
* 不做生产级鲁棒性（但必须有基本防崩溃保护，如除零、NaN 检测）。

---

## 2. 技术栈与运行方式

### 2.1 后端（Python）

* Web 框架：FastAPI（或 Flask，FastAPI 优先）
* 数值计算：NumPy（必需）
* 卷积实现（二选一）：

  * 方案A：PyTorch（CPU）用 `conv2d` 做卷积（推荐，写起来最短）
  * 方案B：OpenCV / SciPy 做卷积与距离变换（可选）
* 图像读写：Pillow（PIL）
* （可选）OpenCV 用于边缘提取/距离变换/加速

### 2.2 前端（HTML + 原生 JS）

* 单页应用：`static/index.html` + `static/app.js` + `static/style.css`
* 可视化：

  * `<canvas>` 显示原图与轮廓叠加
  * `<canvas>` 或 `<img>` 显示 mask / (f_1,f_2)
  * 能量曲线：用 `<canvas>` 手写简单折线图（离线可用，避免 CDN）
* 与后端通信：Fetch API（JSON + base64 PNG）

### 2.3 启动方式（验收要求）

* `python app.py` 启动后端
* 浏览器访问 `http://127.0.0.1:8000/` 打开页面
* 不依赖外网资源（离线可用）

---

## 3. 数学定义（实现必须按此一致）

### 3.1 卷积与核

* 定义离散卷积（与实现一致即可，关键是核对称时前后一致）
  [
  (K_\sigma * u)(\mathbf x)=\sum_{\mathbf y} K_\sigma(\mathbf y-\mathbf x),u(\mathbf y)
  ]
* 高斯核（离散归一化）：
  [
  K_\sigma(\mathbf r)\propto \exp\Big(-\frac{|\mathbf r|^2}{2\sigma^2}\Big),\qquad \sum_{\mathbf r}K_\sigma(\mathbf r)=1
  ]
  核大小建议：`ksize = 2*ceil(3*sigma)+1`

### 3.2 平滑 Heaviside/Dirac

[
H_\epsilon(z)=\frac12\Big(1+\frac{2}{\pi}\arctan\frac{z}{\epsilon}\Big)
]
[
\delta_\epsilon(z)=\frac{1}{\pi}\frac{\epsilon}{\epsilon^2+z^2}
]
实现中 `epsilon` 可调，默认 1.5（像素尺度）。

### 3.3 局部拟合函数（LBF）

[
f_1=\frac{K_\sigma*(I,H_\epsilon(\phi))}{K_\sigma*H_\epsilon(\phi)+\varepsilon_0},\qquad
f_2=\frac{K_\sigma*(I,(1-H_\epsilon(\phi)))}{K_\sigma*(1-H_\epsilon(\phi))+\varepsilon_0}
]
其中 (\varepsilon_0) 是极小值（如 (10^{-6})）避免除零。

### 3.4 全局拟合常数（GBF）

[
c_1=\frac{\sum I,H_\epsilon(\phi)}{\sum H_\epsilon(\phi)+\varepsilon_0},\qquad
c_2=\frac{\sum I,(1-H_\epsilon(\phi))}{\sum (1-H_\epsilon(\phi))+\varepsilon_0}
]

### 3.5 LBF 的局部误差密度（用于 (\phi) 更新）

定义
[
e_1(\mathbf x)=\sum_{\mathbf y}K_\sigma(\mathbf y-\mathbf x),\lvert I(\mathbf x)-f_1(\mathbf y)\rvert^2,\quad
e_2(\mathbf x)=\sum_{\mathbf y}K_\sigma(\mathbf y-\mathbf x),\lvert I(\mathbf x)-f_2(\mathbf y)\rvert^2
]
要求用卷积高效实现（至少 O(N) 次卷积级别，不要显式双重循环）。

推荐展开为（假设核归一化）：
[
e_1 = I^2 - 2I,(K_\sigma*f_1) + (K_\sigma*f_1^2),\qquad
e_2 = I^2 - 2I,(K_\sigma*f_2) + (K_\sigma*f_2^2)
]

### 3.6 曲率与距离正则

* 曲率：
  [
  \kappa=\mathrm{div}\Big(\frac{\nabla\phi}{|\nabla\phi|+\eta}\Big)
  ]
* 距离正则项的驱动：
  [
  R(\phi)=\mathrm{div}\Big(\frac{|\nabla\phi|-1}{|\nabla\phi|+\eta}\nabla\phi\Big)
  ]
  (\eta) 取 (10^{-8})。

### 3.7 (\phi) 的更新方程（Coupled）

[
\phi \leftarrow \phi + \Delta t\cdot
\Big(
-\delta_\epsilon(\phi)\big[(e_1-e_2)+\omega((I-c_1)^2-(I-c_2)^2)\big]
+\nu,\delta_\epsilon(\phi),\kappa
+\mu,R(\phi)
\Big)
]

---

## 4. 算法流程（必须实现）

### 4.1 主循环（ALT 交替最小化模式，默认）

每次迭代 (k=1..N)：

1. 计算 (H=H_\epsilon(\phi))，(\delta=\delta_\epsilon(\phi))
2. 更新 (f_1,f_2)（用卷积闭式解）
3. 更新 (c_1,c_2)（全局均值）
4. 计算 (e_1,e_2)（卷积方式）
5. 计算 (\kappa, R(\phi))（有限差分/卷积核差分均可）
6. 按 3.7 更新 (\phi)
7. 每隔 `viz_stride` 步：

   * 生成 mask：`mask = (phi>0)`（或 <0，需与显示一致）
   * 从 (\phi) 提取零水平集轮廓（推荐：对 `mask` 做边缘检测得到一像素轮廓）
   * 记录能量 (E)（可选）

### 4.2 E2E 模式（可选）

同上，但允许梯度通过 (f_1,f_2,c_1,c_2) 回传。
后端需提供参数 `mode in {"alt","e2e"}`。

---

## 5. 后端 API 设计（FastAPI 示例）

### 5.1 `GET /`

* 返回 `static/index.html`

### 5.2 `POST /api/run`

一次性运行并返回若干帧（简单可靠，推荐）

* 请求：`multipart/form-data`

  * `image`: 上传图片文件
  * `params`: JSON 字符串，包含：

    * `model`: `"lbf" | "gbf" | "coupled"`
    * `sigma, ksize, epsilon, nu, mu, omega, dt`
    * `iters, viz_stride`
    * `init_type`: `"circle" | "rect" | "canvas"`
    * `init_params`: 半径/边距等
    * `mode`: `"alt" | "e2e"`
* 响应：JSON

  * `frames`: 长度 M 的数组，每个元素：

    * `iter`: 迭代数
    * `overlay_png_base64`: 原图 + 轮廓叠加图
    * `mask_png_base64`: 二值 mask
    * `f1_png_base64`, `f2_png_base64`（可选）
    * `stats`: `{c1,c2, energy}`（能量可选）
  * `final`: 同上（最后一帧）
  * `timing_ms`: 总耗时

### 5.3 （可选进阶）`POST /api/step`

用于前端实时播放：每次返回下一步结果与新的 (\phi)。
但这需要传输 (\phi) 或服务端保存 session，复杂度更高；本 SPEC 不强制。

---

## 6. 前端 UI 交互（必须实现）

### 6.1 页面布局

* 左侧：参数面板

  * 上传图片按钮
  * 模型选择：LBF / GBF / Coupled
  * 初始化选择：circle / rect / canvas(可选)
  * 参数滑条（显示数值）：

    * `sigma`、`epsilon`、`nu`、`mu`、`omega`（coupled 时启用）
    * `dt`、`iters`、`viz_stride`
  * Run 按钮
* 右侧：可视化区（至少三块）

  1. 原图 + 轮廓叠加（Canvas）
  2. mask（Canvas 或 img）
  3. （可选）(f_1,f_2) 小图
  4. （可选）能量曲线小图

### 6.2 运行与播放

* 点击 Run 后：

  * 显示加载状态（spinner / 进度条）
  * 收到 `frames` 后：

    * 提供迭代滑条（0..M-1）
    * 提供播放/暂停按钮（按固定 fps 切换帧）
* 每帧显示：

  * `iter` 数字
  * `c1,c2`（若 model 包含 GBF）
  * （可选）`energy`

---

## 7. 数值与工程细节（验收要点）

1. **输入预处理**

   * 默认转灰度（`L`），并归一化到 ([0,1])
   * 限制最大边长（如 512），超大图先等比缩放

2. **边界条件**

   * 计算梯度与卷积时采用 replicate/reflect padding，避免边缘奇异

3. **稳定性保护**

   * 所有除法分母 `+eps0`
   * (|\nabla\phi|) 相关 `+eta`
   * 若出现 NaN/Inf：中止并返回错误信息（前端提示“参数过激进”）

4. **性能目标（CPU）**

   * 256×256 图像，100 次迭代，返回每 5 步一帧（约 20 帧），总耗时最好 < 3 秒（不硬性，但要避免 O(N^2) 双循环）

---

## 8. 验收用例（必须通过）

1. **合成二相圆形**：白圆 + 黑背景

   * GBF 应能收敛到圆边界；LBF 也应收敛。
2. **亮度不均匀（渐变背景）**：圆形目标叠加强烈光照梯度

   * 纯 GBF 可能偏移或失败；LBF 或 Coupled 相对更稳。
3. **初始化敏感性展示**：同一图不同 init 半径

   * 分割结果会变化（这是这个模型的特性），但不应数值爆炸。
4. **极端参数保护**：把 `dt` 拉很大

   * 应提示失败而不是页面卡死/后端崩溃。

---

## 9. 代码结构建议（给生成器一个清晰骨架）

* `app.py`：FastAPI 启动与路由
* `seg/levelset.py`：核心算法（LBF/GBF/Coupled）
* `seg/kernels.py`：高斯核、差分核
* `seg/viz.py`：轮廓提取、叠加渲染、base64 输出
* `static/index.html`
* `static/app.js`
* `static/style.css`
* `requirements.txt`
