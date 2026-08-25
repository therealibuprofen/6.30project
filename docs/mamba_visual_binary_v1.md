# Visual fUS Spatial-Mamba Baseline v1

本实验是受控的 Spatial-Mamba baseline/backbone candidate，只回答：在其他主要组件冻结时，用官方 Mamba 替换 Spatial Transformer 是否更适合当前小样本视觉 fUS。它不是 proposed model。

## 冻结的数据与评测

- 直接读取现有 clean4 block，每个样本 `[4,128,501]`，只保留完整 cycle。
- stimulus 为 grating、dot；non-stimulus 为 stop_after_grating、static。
- 固定 9 session、正式 clean4 cycle-level folds、train-fold-only normalization、seed 0/1/2 和 OOF Balanced Accuracy。
- fold manifest 比较统一 dtype 后执行 exact table equality；真实差异立即报错。
- AdamW、lr 1e-3、weight decay 1e-3、batch size 16、固定 40 epoch。没有 validation/early stopping，test 不参与训练或 epoch 选择。
- batch size 只允许 16。正式 task plan 建立前先以 `[16,4,1,128,501]` 做一次 CUDA forward/backward 显存 preflight；若失败则在任何正式 task 开始前停止并报告，不自动降 batch 或继续混合任务。

## 受控模型差异

CNN stem 直接复用已审查 `cnn_factorized_transformer` 的模块：三层共享单帧 CNN（1→16→32→64，3×3、stride 2、padding 1、BatchNorm、GELU）和 `AdaptiveAvgPool2d((8,32))`。Temporal Transformer、length-4 temporal positional embedding 和 classifier 也直接取自同一已审查实现。

唯一主要替换是两层 Spatial Transformer 改为两层官方 `mamba_ssm.Mamba`：`d_model=64, d_state=16, d_conv=4, expand=2`。8×32 feature map 固定以 row-major 展平为 256 token，并加入 label-independent、跨帧共享的 `[1,256,64]` learnable position。

每层用同一个 Mamba block 分别处理正向序列和 reverse 序列。反向输出 reverse 回原空间顺序后，计算 `0.5 * (forward + backward)`，再作 pre-norm residual 更新。这是 `bidirectional shared-weight spatial scan`，避免额外方向参数和单一 row-major 方向偏置。没有第二种 scan、Mamba2/Mamba3 或架构搜索。

## 可选依赖

普通项目 import 不要求 `mamba_ssm`。只有实例化或训练 Spatial Mamba 时才检查依赖；缺失时明确报错：

```text
Mamba dependency is not installed. Use the dedicated server Mamba environment.
```

`requirements_mamba.txt` 只记录专用 Linux/CUDA 环境的建议官方包版本，不自动安装、不修改基础 requirements。runner 在 `environment.json` 保存实际安装版本。代码始终使用 `from mamba_ssm import Mamba`，不调用 Mamba2/Mamba3。

## 参数和结果完整性

每个 task 保存 total、CNN stem、Spatial Mamba、Temporal Transformer、classifier 参数量；同时报告已审计 Transformer 的 127010 参数及差值，不为参数匹配修改 d_model。

run fingerprint 包含冻结配置、git commit、模型和 runner SHA256，并对所有传递项目源码保存 SHA256。task fingerprint 再加入 session/model/seed/fold/样本数/config/batch size。每个 fold 原子写盘，最后才写 `COMPLETE.json`；汇总前严格复验 fingerprint、身份、配置、参数分项、normalization、40-epoch history、预测、重算指标和混淆矩阵。

正式汇总只读合并旧 clean4、SBIND 和 Factorized Transformer 结果，不重新训练旧 baseline。主要 exact two-sided sign-flip 比较固定为 Spatial Mamba vs CNN mean-pool、Factorized Transformer、SBIND-adapted。

本地没有官方 CUDA 依赖时，Mamba-specific forward/backward 测试必须明确 skip。Mac 不安装依赖、不运行正式实验；第一版服务器结果完成后停止，不自动扩展模型。
