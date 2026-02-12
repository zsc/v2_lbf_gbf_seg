作为分割界的老古董，变分方法\*基于局部二进拟合(Local-Binary-Fit\*)和全局信息拟合的图像分割算法，虽然在今天看来这些算法不具有泛化能力，而且对于图片亮度不一致等问题表现很差，但是其简约的思想和朴素的华丽的数学内涵使得其仍然有一定的实用价值，我在 bilibili 上看到有文章介绍这种耦合局部与全局能量泛函\*的分割算法，就重新写了一下，补充其中关于变分求导的内容：

$$
\begin{aligned}
E^{LBF}(\phi) &= \int d\mathbf{x} \int K_\sigma(\mathbf{y}-\mathbf{x}) |I(\mathbf{x}) - f_1(\mathbf{y})|^2 H(\phi(\mathbf{x})) \\
&+ K_\sigma(\mathbf{y}-\mathbf{x}) |I(\mathbf{x}) - f_2(\mathbf{y})|^2 [1 - H(\phi(\mathbf{x}))] d\mathbf{y} + \\
&\nu \int |\nabla H(\phi)| d\mathbf{x} + \frac{\mu}{2} \int (|\nabla \phi| - 1)^2 d\mathbf{x}
\end{aligned} \quad (1)
$$

这是LBF的能量泛函，而全局信息拟合的为：

$$
\begin{aligned}
E^{GBF} &= \int (I(\mathbf{x}) - c_1)^2 H(\phi(\mathbf{x})) d\mathbf{x} + \int (I(\mathbf{x}) - c_2)^2 [1 - H(\phi(\mathbf{x}))] d\mathbf{x} \\
c_1 &= \frac{\int I(\mathbf{x}) H(\phi(\mathbf{x})) d\mathbf{x}}{\int H(\phi(\mathbf{x})) d\mathbf{x}}, c_2 = \frac{\int I(\mathbf{x}) [1 - H(\phi(\mathbf{x}))] d\mathbf{x}}{\int [1 - H(\phi(\mathbf{x}))] d\mathbf{x}}
\end{aligned} \quad (2)
$$

(2)中由于 $c_1, c_2$ 是标量，因此可以直接得到关于其关于水平集符号函数\* $\phi$ 的极小值，而(1)就需要一点技巧：

首先引入引理：

$$
\int_{-\infty}^{+\infty} (f * g) \times h dx = \int_{-\infty}^{+\infty} (f * h) \times g dx \quad (3)
$$

也即一维卷积的一个性质，因为：

$$
LHS = \int_{-\infty}^{+\infty} h(t) dt \int_{-\infty}^{+\infty} f(t-\tau) g(\tau) d\tau = \int_{-\infty}^{+\infty} g(\tau) d\tau \int_{-\infty}^{+\infty} f(\tau-t) h(t) dt = RHS \quad (4)
$$

这个性质可以验证对于二维卷积也适用，但是前提条件是 $f(\mathbf{x})$ 满足中心对称，一般是卷积核，那么：
(1)中能量可以写成：

$$
\begin{aligned}
J(f_1, \phi(\mathbf{x})) &= \int d\mathbf{x} \int K_\sigma(\mathbf{y}-\mathbf{x}) |I(\mathbf{x}) - f_1(\mathbf{y})|^2 H(\phi(\mathbf{x})) \\
&= \int [K_\sigma * I^2(\mathbf{x})] H(\phi(\mathbf{x})) - 2 [K_\sigma * f_1(\mathbf{x})] I(\mathbf{x}) H(\phi(\mathbf{x})) \\
&\quad + [K_\sigma * f_1^2(\mathbf{x})] H(\phi(\mathbf{x})) d\mathbf{x} \\
\overset{Lemma.1}{=}& \int [K_\sigma * I^2(\mathbf{x})] H(\phi(\mathbf{x})) - 2 \{K_\sigma * [I(\mathbf{x}) \times H(\phi(\mathbf{x}))]\} \times f_1(\mathbf{x}) \\
&\quad + [K_\sigma * H(\phi(\mathbf{x}))] \times f_1^2(\mathbf{x}) d\mathbf{x}
\end{aligned} \quad (5)
$$
$$
\begin{aligned}
\frac{\delta J}{\delta f_1} &= 2 f_1(\mathbf{x}) [K_\sigma * H(\phi(\mathbf{x}))] - 2 \{K_\sigma * [I(\mathbf{x}) \times H(\phi(\mathbf{x}))]\} = 0; \\
&\Leftrightarrow f_1(\mathbf{x}) = \frac{\{K_\sigma * [I(\mathbf{x}) \times H(\phi(\mathbf{x}))]\}}{[K_\sigma * H(\phi(\mathbf{x}))]};
\end{aligned}
$$

同理可以得到：

$$
f_2(\mathbf{x}) = \frac{K_\sigma(\mathbf{x}) * \{[1 - H(\phi(\mathbf{x}))] I(\mathbf{x})\}}{K_\sigma(\mathbf{x}) * [1 - H(\phi(\mathbf{x}))]} \quad (6)
$$

对于多卷积核也可以同样的方法求解出 $f_1, f_2$ ，那么对于总的能量函数：

$$
E(\phi(\mathbf{x})) = E^{LBF}(f_1(\mathbf{x}), f_2(\mathbf{x}), \phi(\mathbf{x})) + \omega E^{GBF}(c_1(\mathbf{x}), c_2(\mathbf{x}), \phi(\mathbf{x})) \quad (7)
$$

就可以求对于水平集符号函数的变分导数了，最后使用梯度下降法\*求解，我这里实现了使用pytorch\*自动求导求解该问题，需要注意的是，这种方法需要很细致地手动调节迭代次数，滤波程度以及水平集初始值，所以在深度学习盛行的今天已经没啥参考价值了。

---
细说以上，然后生成一个可由 codex / gemini-cli 实现的 python+html demo SPEC 中文 markdown，包含数学公式。
