# Visual fUS Lightweight Spatiotemporal Transformer Baseline v1

本实验只加入一个冻结的 `cnn_factorized_transformer` baseline，用于检验 CNN 局部空间编码之后的全局空间 self-attention 与短时间 self-attention。它不是 proposed model，也不声称是方法创新。

## 冻结协议

- 数据直接读取现有 clean4 block：每个样本为 `[4,128,501]`，只保留完整 cycle。
- 二分类标签固定为 stimulus（grating、dot）与 non-stimulus（stop_after_grating、static）。
- 9 个 session、cycle-level grouped folds、正式 clean4 fold manifest、seed 0/1/2 均不改变。
- manifest 比较先规范 session、fold、cycle list、计数及 JSON class counts 的 dtype，再进行逐表 exact equality。这样避免 CSV 将 `test_cycles="0"` 推断为整数造成假 mismatch，同时真实内容变化仍会报错停止。
- normalization 固定为 arcsinh 后逐像素 z-score；均值和标准差只从 outer train fold 的全部四帧估计，test fold 只接受变换。
- OOF Balanced Accuracy 是主指标。

## 模型

三层共享单帧 CNN 为 `1→16→32→64`，每层使用 3×3、stride 2、padding 1、BatchNorm 与 GELU；`AdaptiveAvgPool2d((8,32))` 将每帧变成 256 个 64 维空间 token。learnable 行、列 embedding 共同构成二维空间位置编码，不含标签信息。

同一个两层 spatial Transformer 处理所有四帧，参数不会按帧复制。空间 token mean-pool 后得到四个 64 维帧表示，再加入长度 4 的 learnable temporal position，经过一层非 causal temporal Transformer，时间 mean-pool 后由 `LayerNorm(64) → Dropout(0.25) → Linear(64,2)` 分类。

clean4 只有四个时间点，因此没有必要把全部 1024 个时空 token 组成昂贵的全局 attention。factorized baseline 分别建模同帧远距离空间依赖和四个有序 fUS 时间点之间的依赖。

## 训练与 epoch 选择

AdamW、lr 1e-3、weight decay 1e-3、batch size 16、40 epoch。与现有正式 clean4 deep benchmark 一致，不建立 validation fold、不 early-stop，test fold 从不用于训练或选择 epoch。`overfitting_summary.csv` 因此将 validation 标为不适用；`best_epoch` 只是训练曲线诊断，正式 `selected_epoch` 始终为 40。

若 RTX 3080 10GB OOM，只允许所有 session 统一将 batch size 从 16 降为 8，再必要时降为 4。batch size 属于 fingerprint；不同 batch size 的任务不能混合续跑。

## 完成态与防混合

run fingerprint 同时包含冻结实验配置、git commit、模型实现版本、模型源码 SHA256 和 runner 源码 SHA256。task fingerprint 再加入 session/model/seed/fold/测试样本数。代码或配置变化后，旧任务不会被视为当前任务。

每个 fold 完成后原子写盘，最后才写 `COMPLETE.json`。正式汇总前对每个 task 重新严格验证 fingerprint、身份、模型配置、normalization、40-epoch history、预测概率、由预测重算的指标和混淆矩阵；不会只检查完成标记。

## 运行边界

本地 sanity 仅限 session 710、fold 1、seed 0、2 或 3 epoch、workers 0，且写入 `sanity/`，不进入正式汇总。正式 9-session 运行只在服务器 CUDA 环境执行。runner 可按 fold/seed 断点续跑并打印已完成任务数/总任务数。
