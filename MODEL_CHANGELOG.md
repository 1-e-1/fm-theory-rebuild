# 模型源码与正确性审查记录

日期：2026-10-09。本轮范围为 SEQ_IID / ROBOT_8DOF 的五个主模型、五个种子；本记录涉及必要正确性检查，不计入 50 个正式配置。

交付版本 `theory_fm_models.py`：1.0.0。原版快照在 `original_snapshot/theory_fm_models.py`。

原版 SHA-256 与实际执行 SHA-256 均为：

`fd97ebc960271869c4906cf4ad8f67252e30ebb39b84f461cc882442a2c76fbe`

**模型源码修改：无；原版与实际执行文件逐字节相同，统一 diff 为空。** 未改变模型方程、时间轴、势函数、阶数域、扩散位置、标签或 VDC 双势结构。未引入循环网络、隐藏状态、attention、AMP、TF32、自定义算子或性能重写。

模型审查已实际读取 `MODEL_GUIDE.md`、`CODEX_RUNBOOK.md`、`START_CODEX_PHASE1_50_RUNS.txt` 与模型完整源码。理论 PDF 与定义的核对由主执行代理完成并记录在项目来源/进度中。

## 实际测试

命令：

```powershell
python tests/model_correctness.py
python tests/model_benchmark.py
python -m pytest tests/test_experiment_integration.py
```

`tests/model_correctness_report.json` 记录 23/23 PASSED、输入、误差、dtype、设备、源哈希与测试哈希。测试只使用算子/空间场输入，不读取正式训练或测试数据，不导入 checkpoint，不运行任何额外研究模型的训练。

已通过：

- CPU float64 / complex128：Lanczos Gamma 对 SciPy 独立参考最大误差 4.45e-15；实/虚参数梯度对中心差分最大误差 3.17e-10。
- RL 算子：alpha=0、alpha=-1、一般复阶常数和线性 PL 历史；独立 PC cell 积分参考；保留非零 f0 初始项与首 cell。`t^2` 连续 Gamma 参考随 H=17/65/257 加密误差下降，PL 与 PC 按各自定义比较。
- 阶数实部、虚部、系数核梯度：CPU64 最大中心差分误差 1.99e-10，GPU32 最大误差 1.17e-5。
- 全五模型 CPU64/GPU32 的 D=64 前后向、所有参数有限梯度、同输入复现、forward 不消费 RNG、非零首 cell、no_grad 推断与 inference_mode 明确拒绝。
- 势函数 W 偶性、空间梯度有限差分、含权重源粒子置换不变性、固定 query 历史影响、alpha=0 历史退化。
- query/source 共底层状态时的固定概率场 partial_x；外层 source/history 参数依赖仍保留。
- 完整核实化 qF+rJF 对二维复数编码误差 2.64e-16；故意省略虚部核产生非零差异。
- 相同 state_dict 的 SDE/ODE 漂移完全一致；b=0 退化实阶；第二势恒零退化单势。
- 二维 VDC div=-Delta A / curl=-Delta B 误差 4.45e-16；D=4 全部六个反对称分量对直接 Jacobian 有限差分，指定 pairs 与正交 probes trace 核对。
- 嵌套势梯度的 force-loss 对网络与阶数参数中心差分：CPU64 最大绝对误差 9.34e-11；GPU32 最大绝对误差 1.25e-4（rtol=0.002 与 float32 差分误差容限内）。
- D=64 J 的反对称与 J^2=-I、序列可逆 flatten、32维/64点共享机器人映射端点与解析端点导数；非法模型ID/维度/时间/条件/history/dtype 明确拒绝。
- 独立解析 Gaussian 教师验证 v+kappa*score 的 FP 符号，最大残差 1.25e-10；Brownian 两个 h 的方差为 2*kappa*h，跨坐标经验相关小于 0.0046。
- 独立 reference 输入与当前 query 不共享配对端点/噪声，reference 的 Y/epsilon 跨历史节点保持一致。

真实 experiment 训练器/采样器集成证据独立记录在 `tests/experiment_integration_results.txt`；模型层参考测试不替代集成测试。

## 测试首次失败与复测

首次测试 20/21 PASSED。失败发生在测试用独立 PC 参考的 Python `0 ** complex` 表达式，抛出 ZeroDivisionError；交付模型对应测试输入未报错。参考实现按 Re(p)>0 的数学定义显式设置零基数幂为 0 后复测通过。首轮原始报告保留在 `tests/model_correctness_report_attempt1.json`。该修正只修改测试，不涉及模型源码。

## 成本检查

`tests/model_benchmark_results.json` 是资源检查：D64、E1、Q64、N32、H17、hidden128、3层、GPU float32（核 complex128），每模型 3 次前后向、0 次优化更新。它不是短 pilot 或正式训练。

预热后每次 forward+backward：STANDARD 约 0.001 秒；REAL/COMPLEX 约 0.075–0.086 秒；VDC 约 0.116–0.122 秒。模型分配峰值约 0.387 GB，VDC 约 0.615 GB。仅该测量不包含优化器、validation、数据准备、日志或自主采样，应以正式 run 的实测开销为准。

## 同一 checkpoint 附加检查

新增独立 `experiment/diagnostics.py`，未修改交付模型。它在每个正式 checkpoint 上只做附加评价：固定 validation query4、独立 N32/H17 reference、P4 Rademacher 散度迹估计、预定四个高维 curl pairs；另做两个独立 ensemble 的 32/64/128 步与 N32/64/128 加密检查。共有 N128 初值与 128 步 Brownian 细增量，粗增量用细增量求和。原始浮点数组、差值 RMS、单独开销全部保存，checkpoint 参数/配置哈希变化时拒绝沿用旧附加检查。

这些是无真值场诊断和有限数值参考，不报告预测准确率，不把最细步/最大 N 当精确解，不拟合未经支持的收敛率，不计为新的训练配置。必要接口检查 `tests/model_diagnostics.py` 两项通过。

本项目新生成的 COMPLEX_ODE pilot checkpoint 上实际运行全部五项唯一采样检查，训练更新数为 0；总成本 42.60 秒，其中 validation 场诊断 1.11 秒。完整 raw arrays 位于该 pilot 的 `checkpoint_diagnostics/`；成本记录为 `tests/model_diagnostics_pilot_benchmark_result.json`。此记录不使用旧工程 checkpoint，也未访问正式 test。
