# 从理论稿重建的五类流场模型：定义、接口与理论范围

版本：`fm_theory_rebuild_v1 / 1.0.0`。

交付是一个完整模型定义文件 `theory_fm_models.py`，不是训练工程。代码没有训练循环、数据下载、状态 ODE/SDE 求解器或自动实验。交付前仅通过 AST 静态语法解析；没有执行模型前向、反向、数值测试或训练。Codex 必须先完成任务书中的实际检查。

## 1. 依据是什么，哪些是本次新设计

主要来源是用户的59页未发表研究稿：

> 《复阶场记忆驱动的 McKean–Vlasov 动力学：适定性、粒子极限与实密度谱》，完整理论研究稿1.0，2026-10-07。

原 PDF 的 SHA-256：
`04cd5eb0d681f27db7f2b29a82f88341ca4543ccbe6f079887033ce809aea2ad`。

本次复读并按以下原文构造，不按前期 Caputo 状态模型构造：

| 原文位置 | 本次保留的内容 |
|---|---|
| 第6–8页，§2.1–2.4 | 实值概率泛函；实偶势 W；复数编码速度；固定空间位置的历史算子；普通 Itô SDE |
| 第10–13页，§3.1–3.3 | 因果 Riemann–Liouville 分布算子，保留真实初始项 |
| 第19–21页、§7 | 均值场与 Brownian 经验场正则性不同；有限粒子主要范围 Re α<1/2 |
| 第23–29页，§4–6 | 在所列条件下的适定性、参数扰动稳定性；不是生成质量保证 |
| 第38–39页，§8.1 | 完整核实化：qF+rJF；不能只实化系数而漏掉核的虚部 |
| 第48–53页 | 虚部不自动提高稳定性；积分记忆不自动保持当前自由能下降 |

**本次自行定义、并不在原文定理内的内容：**

1. 用平滑前馈 MLP 参数化势函数并接入 FM 回归目标。
2. 偶数维 `D>2` 的固定复结构 J 扩展，用于完整序列/轨迹系数向量。
3. 双概率势的速度—散度—旋度（VDC）构造。
4. κ=0 的确定性消融、实际离散器、机器学习泛化与任意拟合残差的输送误差。
5. 阶数在优化步骤之间学习；一次生成过程中固定。原稿没有随机阶数过程。

原文和《原论文方法验证与学习接口进展》的数值结果只支持其各自写明的范围。这里不会将旧实验换标签作为新模型结果。

## 2. 五个主模型：请使用完整名称

旧实验的 M0–M3 已经经历过多次含义变更。本包拒绝接受 `M0` 等短别名，避免混用旧数据。

| 顺序 | 唯一 ID | 场的构造 | 状态 Brownian | 阶数 |
|---|---|---|---|---|
| 1 | `FM_STANDARD` | 前馈网络直接输出速度 | 无 | 无复阶场算子 |
| 2 | `FM_REAL_ODE` | 单概率势 + 实数阶场响应 | 无 | a 可学习，b=0 |
| 3 | `FM_COMPLEX_SDE` | 单概率势 + 复数阶场响应 | 有 | a、b 可学习 |
| 4 | `FM_COMPLEX_ODE` | 与第3项相同的漂移结构 | 无 | a、b 可学习 |
| 5 | `FM_COMPLEX_VDC_ODE` | 双概率势 + 复阶响应 + 一致的速度/散度/旋度 | 无 | a、b 可学习 |

第5项默认 κ=0，用来与第4项单独比较空间场参数化。`FM_COMPLEX_VDC_SDE` 已实现为补充对照，用来与第3项比较。不能把主表中的差异同时归因于两个不同改动。

随机性仅指生成状态中的 Brownian 增量；所有模型的初始噪声、训练采样、参数初始化仍然可以随机。“无随机过程模型”不等于输出确定的一条固定轨迹。

## 3. 统一的“无记忆网络”含义

所有可学习网络均为 `Linear → Tanh` 前馈网络，默认宽128、隐藏层3层。没有 RNN、GRU、LSTM、时序 attention、KV cache、SSM、循环隐状态或 dropout。主干基元一致，但不同输出形式/网络数量导致实际参数量不同，必须实测并报告。

复阶系统本身仍然有**显式数学历史**。该历史是模型调用的输入 `FieldHistory`，不藏在神经网络中。

如果把数学历史也删除，得到的就不再是原论文的场记忆模型。

## 4. 三个不同的轴

- `tau`：噪声到目标分布的生成时间，原理论的 t 在应用中对应这个轴。
- `s`：一条输入时序的观测时间/序列索引，或机器人轨迹的执行时间。
- `x∈R^D`：一个完整样本在生成空间中的位置。

例：`[64,3]` 时序通过可逆 reshape 变成 `D=192` 的一个样本。复阶算子作用于 `F[rho_tau](x)` 随 tau 的历史，不对该时序的64个观测位置直接卷积。

因此本方法的生成历史先验是否改善观测时间上的长依赖，是要做实验的问题，不是数学定义自动给出的结论。

## 5. 数学定义

### 5.1 标准 FM

\[
 b_\theta(x,\tau,c)=\mathrm{MLP}_\theta(x,\tau,c),\qquad dX_\tau=b_\theta\,d\tau.
\]

这是普通一阶状态 ODE 的神经速度场，没有人为加局部卷积、势函数、FIR或算子分支。时间只作前馈编码。

### 5.2 概率势与基础场

对固定条件 c，构造

\[
 \mathcal E_\theta[\rho\mid c]
 =\int V_\theta(x,c)\rho(dx)
 +\frac12\iint W_\theta(x-y,c)\rho(dx)\rho(dy).
\]

\[
 U_\theta[\rho](x,c)=V_\theta(x,c)+(W_\theta(\cdot,c)*\rho)(x),
 \quad F_\theta[\rho]=-\nabla_x U_\theta[\rho].
\]

V/W 各用一个 Tanh MLP。势输出经 tanh 限幅并乘 Gaussian 空间包络。W 将 z 与 -z 的网络输出平均，保证偶性。对每组有限参数和固定条件，这种设计给出平滑有界 V，以及快速衰减的平滑 W；它同时限制了表达能力，不是任意空间势的完整表示。

V/W 不接收 tau；只由 rho 的演化和时间算子产生时间依赖。这与原稿固定势的结构一致。不得自行添加任意时间漂移残差来掩盖拟合不足。

### 5.3 统一复阶响应

\[
 \mathcal K_\Theta f=\sum_k c_k T_*^{\alpha_k}{}_0D_\tau^{\alpha_k}f,
 \qquad \alpha_k=a_k+i b_k.
\]

代码默认4个谱节点，实部可跨0：

\[
 -1<a_k<0.4,\qquad |b_k|<0.5.
\]

这些是数值实验域，不是否认整个复平面的分布定义。Brownian 粒子定理的阈值必须保留。配置允许将负下限扩大，但禁止将正上限悄悄改到1/2以上。

主组的 c_k 是相同参数化的有界实系数。这样比较 M1与复阶模型时，不把“自由复系数旋转”混入“复阶虚部”。`FM_REAL_ROT_ODE` 单独允许复系数。所有阶数和系数一次生成内固定，训练更新时才变化。

对完整时间核写 q=Re K、r=Im K。D=2时

\[
 b_\Theta=q*F+r*JF,
 \quad J=\begin{pmatrix}0&-1\\1&0\end{pmatrix}.
\]

`FM_REAL_ODE` 中 r=0；`FM_COMPLEX_ODE/SDE` 保留 q、r 两项。SDE与ODE在相同参数、相同历史、相同查询点上返回完全相同的漂移；只有扩散系数不同。

### 5.4 双势 VDC 模型

再引入另一个独立的概率势 S[ρ]，与 U 使用相同形式但不同输出通道，定义基础场

\[
 F_{\rm VDC}[\rho]=-\nabla U[\rho]-J\nabla S[\rho].
\]

对它施加相同复阶响应，得

\[
 b=-\nabla A-J\nabla B,
 \quad A=q*U-r*S,
 \quad B=r*U+q*S.
\]

D=2 时：

\[
 \operatorname{div}b=-\Delta A,\qquad
 \operatorname{curl}b=-\Delta B.
\]

这是**势函数式 Hodge 构造**：网络定义两个势，速度由它们的梯度组成，散度和旋度由同一个速度计算。没有三个互相不一致的独立预测头。它不是先任意预测两个数就把它们直接当速度，也没有用 rho+epsilon 的除法篡改概率流。

与“先输出散度和旋度再解Poisson”相比，这是本次明确选择的另一种构造：避免高维网格Poisson求解，同时保证几何一致性。不要给它改回二维FFT硬套高维。

本版本势具有空间衰减，选择了无额外 harmonic/constant-flow 分支的结构子类。并未声称在任意有界区域上完成给定边界条件的唯一 Hodge 反演。

### 5.5 高维扩展

当 D=2m，用固定 J 满足

\[
 J^T=-J,\quad J^2=-I.
\]

默认按相邻坐标建立正交二维块，但这仅是指定的坐标几何，不是给某些真实数据通道贴“物理虚部”标签。所有真实坐标都保留，输出也是 D 维实向量。不截掉一半坐标，不用复值概率。

对高维速度应记录

\[
 \Omega_{ij}=\partial_i b_j-\partial_j b_i,
\]

它是反对称张量，不是一个标量。代码允许指定坐标对，只计算相应分量。D>2时两个势加固定J是一个受限的构造，**不覆盖所有高维旋度张量**；新增的高维、双势及条件版本都不能直接引用二维原稿定理作为保证。

必须做固定J换基/配对敏感性，排除优势仅来自某一种坐标配对。

## 6. 随机过程和训练目标

本包只返回 `diffusion=sqrt(2*kappa)`；Brownian增量必须由外部求解器产生。模型 forward 不抽随机数，不更新alpha，不推进状态。

ODE速度满足连续性方程；正扩散SDE需要匹配完整漂移

\[
 b^*=v^*+\kappa\nabla\log\rho_\tau^*.
\]

采样时输出已是完整漂移，不再添加一次score。不要把 Brownian 增量除以 h 当监督目标。

实验任务书使用共同平滑Gaussian条件路径：

\[
 X_\tau=\tau Y+\sigma(\tau)\epsilon,
 \quad \sigma=1-(1-\sigma_{\min})\tau.
\]

条件速度和score目标为

\[
 u=Y+\sigma'\epsilon,\quad s=-\epsilon/\sigma.
\]

ODE回归u；SDE回归u+κs。正确条件平均对应同一边缘概率路径。各组共享sigma_min，终点是数据的轻微Gaussian平滑；不能把σmin=0.05称为精确原始经验分布，也不能只给某组额外去噪。

参考群历史必须独立于当前查询样本的端点Y/噪声epsilon。否则可以泄漏查询样本答案，破坏FM回归的解释。条件任务中每个参考群只包含同一条件下的样本。

训练用参考历史；采样用自己的历史。二者误差传递不是原稿已经证明的任意神经网络泛化定理。需要分别评价teacher-forced漂移残差和自主生成结果。

### 一个决定旋度实验是否公平的重要事实

对上述各向同性Gaussian条件路径，在tau>0可直接推导

\[
 v^*(x,\tau)=\frac{x}{\tau}+\frac{\sigma(\tau)}{\tau}\nabla\log\rho_\tau^*(x).
\]

所以对任意数据分布，该目标的边缘速度都是梯度场；加空间常数κ的score后仍是梯度场。这不仅是某个GMM的特例。不能只在这类目标上看到虚部趋0，就断言复阶数学不成立；也不能预设旋度分支一定提高表现。

主性能实验保留这个公认的普通路径。另设计明确标为“机制”的旋转参考路径：

\[
 Z_\tau=R_\tau X_\tau,\quad R_\tau=\exp(\theta(\tau)J),
 \quad\theta(0)=\theta(1)=0.
\]

其目标为

\[
 v_Z^*(z,\tau)=\theta'(\tau)Jz+R_\tau v_X^*(R_\tau^Tz,\tau).
\]

这提供真实非梯度成分，且端点分布不变。全部模型使用同一路径；此实验衡量旋转场表示，不能把人为选择的旋转路径优势包装成原始标准路径上的普遍SOTA。

## 7. 接口

依赖：Python标准库与PyTorch。兼容性仍需在用户实际环境检查；优先使用现有隔离环境，不要求升级全局软件。

```python
from theory_fm_models import ModelConfig, build_model, FieldHistory

cfg = ModelConfig(state_dim=192, condition_dim=0)
model = build_model('FM_COMPLEX_SDE', cfg).to(device='cuda', dtype=torch.float32)

# 外部准备：x[E,Q,192]，tau一个标量。
# sources[E,H,N,192]；times[H]从0到tau。
history = FieldHistory(times, sources)
out = model(x, tau, history=history)
b = out.drift
sigma_state = out.diffusion
```

`FM_STANDARD` 不需要history：

```python
out = standard_model(x, tau, condition=condition)
```

散度/旋度仅在需要诊断的小批次计算：

```python
fields = model.differential_fields(
    x_small, tau, history=history, condition=condition,
    curl_pairs=[(0, 1), (2, 3)],
    divergence_probes=external_fixed_rademacher_probes,
)
```

- 没有probes时，散度是完整空间Jacobian的精确迹；高维成本较高。
- 有probes时，返回明确标注的随机迹估计，不能伪称逐点精确散度。
- 旋度分量是指定坐标对的精确偏导差，不是整个高维旋度范数。
- 计算空间偏导时固定参考概率历史；不是对整个交互粒子系统状态的全Jacobian求导。
- 采样请使用 `torch.no_grad()`，不要用 `torch.inference_mode()`，势梯度需要临时启用autograd。
- 验证/采样可 `create_graph=False`；训练势参数需要保留梯度图，不能把force或kernel转成NumPy。

## 8. 时间离散接口

`forward(..., t, history)` 返回最后正时间节点的漂移：对每条固定空间历史采用分段线性插值，用RL含初始项的解析积分计算。

`cell_average(x,left,right,history)` 不推进状态，只返回冻结空间查询x时，区间[left,right]上的平均漂移。场模型用分段常值历史，最后一段保持左端场；初始区间使用真实F[rho0]，不把漂移硬置0。普通MLP用3点Gauss时间积分。

第一步如何调用、后续节点如何保持一致，在任务书中规定。不能把这两种不同历史近似的有限步结果声称完全相同。需要实际加密检查。

特别禁止：

- 在t=0直接取发散的正阶点漂移；
- 简单删除初始场；
- `D^alpha(f-f0)`冒充原RL算子；
- 把每个历史时刻不同的接收位置算出来的速度存下后卷积；
- 将κ>0改成随机alpha或OU阶数；
- 仅为了通过测试把复数输出取模；
- 训练后关闭分支代替从头训练的基线。

## 9. 能从定义直接推出的性质

以下是本次模型定义的代数推论，不是实测性能：

1. 相同输入和参数时，complex SDE/ODE的漂移完全相同。
2. 将全部虚部阶数设为0，复阶共享势模型退化为实阶模型。
3. 双势模型的第二个势恒为0时退化为单势模型。
4. 二维中VDC模型的散度/旋度由同一个速度给出，不存在独立头之间的不一致。
5. 相同节点数和损失、忽略训练及正则化差异时，实阶类嵌入复阶类；复阶单势类嵌入双势类。因此最优可实现拟合损失不会因扩大模型类而升高。有限优化、泛化与采样质量没有相同保证。
6. Brownian噪声改变路径律与FP目标；SDE和ODE的不同损失不能直接比较并解释为某一组“学习更好”。
7. 两个实网络也可表示与复数运算相同的实化映射。研究价值在结构化参数化、统计效率和稳定的实际收益，不是“实值网络绝不可能表示复阶效果”。

## 10. 参考资料

[P1] 用户理论稿，见上述文件hash与章节。
[P2] 用户《原论文方法验证与学习接口进展》，特别第1、5–8页。它明确区分完整漂移、原方法、二维pilot与高维未完成内容。
[E1] Lipman et al. Flow Matching for Generative Modeling. https://arxiv.org/abs/2210.02747
[E2] Albergo, Boffi, Vanden-Eijnden. Stochastic Interpolants: A Unifying Framework for Flows and Diffusions. https://arxiv.org/abs/2303.08797
[E3] PyTorch 2.9 Autograd mechanics. https://docs.pytorch.org/docs/2.9/notes/autograd.html
[E4] PyTorch autograd.grad. https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html

外部来源只用于说明FM/随机插值和软件接口。本包新增高维与双势定义不是声称这些论文已经给出的结果。
