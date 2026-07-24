# Monkey Ultrasound Visual-Stimulus Decoding Benchmark

这个项目目前实现了猴子视觉刺激超声数据的单 session 解码基准。目标是先把一个数据集、一种实验范式、几类常用解码方法跑通，并且严格避免训练/测试泄漏；后续可以在同一接口上扩展到多 session、跨 session、公开数据集和更多模型。

## 当前数据

原始数据位于 `data/`：

- `data/708`
- `data/709`
- `data/710`
- 以及后续加入的其他 session 目录

每个 session 由一批 MATLAB v7.3 `.mat` 文件组成。每个 `.mat` 文件实际是 HDF5 容器，核心变量是：

```text
Data_SVD: shape = 128 x 501
```

可以把每个文件理解为一个约 4 秒的超声数据组。当前解码把每个文件当作一个样本。

## 实验范式和标签

`readme.pptx` 中说明了视觉刺激范式：

```text
一个周期 120 s
30 s 光栅
30 s 停止
30 s 光点
30 s 静止
每组超声数据约 4 s
```

因此当前标签是根据文件编号和时间范式推断出来的，不是从原始事件日志读取的。若以后拿到真实刺激时间戳，应优先替换为事件表驱动标签。

当前四个 block：

| Block | 二分类标签 | 刺激类型标签 |
| --- | --- | --- |
| `grating` | `stimulus` | `grating` |
| `stop_after_grating` | `no_stimulus` | 不使用 |
| `dot` | `stimulus` | `dot` |
| `static` | `no_stimulus` | 不使用 |

已实现两个任务：

- `binary`：有无刺激二分类，`stimulus` vs `no_stimulus`
- `stimulus_type`：两种刺激类型二分类，只取 `grating` vs `dot`

## 默认取了几帧

默认启用 `clean_middle`，只取每个 30 秒 block 中间比较干净的帧。当前参数是：

```text
clean_margin_s = 8.0
```

含义是：每个样本用其 4 秒数据组的中心时间贴标签；如果该中心时间距离所在 30 秒 block 的起点或终点不足 8 秒，就丢弃。

由于每组数据约 4 秒，一个 120 秒周期大约 30 组。以一个完整周期为例，默认保留：

| Block | 保留帧数 |
| --- | ---: |
| `grating` | 4 |
| `stop_after_grating` | 4 |
| `dot` | 4 |
| `static` | 4 |
| 合计 | 16 |

因此对 708 使用完整周期裁剪时：

- 共有 6 个完整周期。
- `binary` 任务保留 `6 x 16 = 96` 帧。
- `stimulus_type` 任务只取 `grating` 和 `dot`，保留 `6 x 8 = 48` 帧。

这些样本清单会输出到：

```text
reports/decoding/samples/{session}_{task}_samples.csv
```

例如：

```text
reports/decoding/samples/708_binary_samples.csv
reports/decoding/samples/708_stimulus_type_samples.csv
```

## 完整周期裁剪

在按 cycle 做交叉验证之前，代码会先做完整周期裁剪。当前范式中：

```text
1 个 cycle = 120 s
1 帧/组 = 4 s
所以 1 个完整 cycle = 30 帧
```

默认启用：

```text
trim_incomplete_cycles = true
```

裁剪顺序是：

1. 先应用 `--analysis-limit`，如果用户提供了分析范围。
2. 在该范围内统计每个 cycle 有多少帧。
3. 只保留刚好有 30 帧的完整 cycle。
4. 再做 clean-middle 筛选。
5. 最后按任务过滤类别，例如 `stimulus_type` 只保留 `grating` 和 `dot`。

这样可以避免 709 末尾 7 帧被单独划成一个 partial cycle 后进入 grouped CV。也能统一处理 708、710 和其他 session 的尾部不完整周期。

为什么之前 708 看起来是对的、而 710 曾经是错的：

- 708 原始数据有 184 帧。
- 如果不做任何范围限制，最后 181-184 是一个只有 4 帧的不完整 cycle。
- 完整周期裁剪会自动保留 1-180，丢掉 181-184。
- 之前代码里错误保留了 `DEFAULT_ANALYSIS_LIMITS["710"] = (1, 180)`，导致 710 也被手工截到 1-180，只剩 6 个周期。
- 现在已经取消 708/710 的默认手工范围限制。710 会使用全量 556 帧，先保留 1-540 的 18 个完整周期，再丢掉 541-556 的 partial cycle。

如果想临时保留不完整周期，可以加：

```bash
--no-trim-incomplete-cycles
```

但正式解码默认不建议这样做。

完整周期裁剪信息会写入两个地方：

```text
reports/decoding/summary/{session}_{task}_summary.json
reports/decoding/audit/{session}_{task}_cycle_selection_report.csv
```

此外可以用 audit 脚本检查 `data/` 下所有 session：

```bash
.venv/bin/python scripts/audit_complete_cycles.py
```

输出：

```text
reports/decoding/audit/complete_cycle_audit.csv
```

当前全 session audit 结果：

| Session | Raw Frames | Complete Cycles | Frames Kept Before Clean-Middle | Dropped Frames | Dropped Indices |
| --- | ---: | ---: | ---: | ---: | --- |
| 708 | 184 | 6 | 180 | 4 | 181-184 |
| 709 | 667 | 22 | 660 | 7 | 661-667 |
| 710 | 556 | 18 | 540 | 16 | 541-556 |
| 807 | 360 | 12 | 360 | 0 | none |
| 813 | 324 | 10 | 300 | 24 | 301-324 |
| 817 | 604 | 20 | 600 | 4 | 601-604 |
| 822 | 300 | 10 | 300 | 0 | none |

## 交叉验证

训练/测试划分不随机拆帧，而是按周期分组。

当前实现：

- 分组字段：`cycle`
- 若周期数不超过 `--max-folds`，使用 leave-one-cycle-out。
- 若周期数超过 `--max-folds`，把 cycle 分成最多 10 折。

以 708 的 `1:180` 为例：

- 共有 6 个周期。
- 使用 6 折 leave-one-cycle-out。
- 每折完整留出一个 120 秒周期作为测试集。

这样避免同一周期内相邻时间点同时出现在训练集和测试集里。

## 已实现模型

### PCA + LDA

方法名：

```text
pca_lda
```

每一折内部严格遵守：

1. 只在训练集上计算训练均值。
2. 训练集减去训练均值后做 PCA。
3. PCA 保留 95% 方差，参数可通过 `--pca-variance` 修改。
4. 测试集只减去训练均值，再乘训练集 PCA 主成分。
5. LDA 只在训练集 PCA 特征上训练。
6. 训练好的 LDA 模型预测测试集。

### 固定时间窗 PCA + LDA

统一固定时间窗扫描只运行 `pca_lda`，正式入口是：

```bash
.venv/bin/python scripts/temporal/run_fixed_window_analysis.py \
  --sessions 708 709 710 807 813 817 822 \
  --tasks binary stimulus_type \
  --method pca_lda \
  --window-spec all \
  --clean-margin-s 8 \
  --pca-variance 0.95 \
  --max-folds 10 \
  --output-root results/runs/temporal_windows \
  --run-name fixed_window_unified_v1 \
  --reuse-compatible-results
```

每个样本来自同一 `cycle`、同一 `block_name`、同一标签内的固定连续 K 帧。K 支持 1、2、3、4。窗口配置用 block 内相对位置 `position` 定义：`[0]`、`[1]`、`[2]`、`[3]`、`[0,1]`、`[1,2]`、`[2,3]`、`[0,1,2]`、`[1,2,3]`、`[0,1,2,3]`。同一个 `window_id` 在所有 session 中使用相同 positions；每个 block 对每个固定窗口最多生成 1 个样本。

多帧样本在进入 PCA + LDA 前按时间顺序展平拼接。交叉验证分组仍为 `cycle`；708 有 6 个完整 cycle 时使用 6 折，其他 session 使用现有 `grouped_cv_splits(..., max_folds=10)` 逻辑，最多 10 折。标准化、PCA 和 LDA 都只在训练 fold 上 fit，测试 fold 只使用训练 fold 得到的 transform 和模型。

时间字段是 nominal 映射，不是事件日志的精确时间戳。原因是当前原始数据没有单独事件表；代码按 `.mat` 文件序号和约 4 s 采集组推断标签，并把文件中心时间用于 clean-middle。对 `stimulus_type`，grating/dot 的同一 position 具有相同名义中心，位置 0/1/2/3 约为 10/14/18/22 s，aggregate 中 `time_mapping_status = nominal`。对 `binary`，no-stimulus block 与 stimulus block 的 4 s 采集组相位不同：同一 position 在 grating/dot 与 stop/static 之间约有 2 s 差异，因此 aggregate 中 `time_mapping_status = block_dependent_nominal`，不再把不同 block 的时间简单平均成唯一时间；主汇总保留 `position` 和 block-specific 时间列表，完整映射写入 `aggregate/fixed_window_block_time_mapping.csv`。

统一输出位置：

```text
results/runs/temporal_windows/fixed_window_unified_v1/
  {session}/{task}/{window_id}/
  aggregate/
```

### cPCA + LDA

方法名：

```text
cpca_lda
```

这里实现的是类别对比 PCA 版本：

1. 先在训练集上做普通 PCA 预降维，避免小样本高维协方差不稳定。
2. 在预降维空间中计算类间差异和类内差异。
3. 提取更强调类别差异的投影轴。
4. 再用 LDA 在训练集上训练并预测测试集。

该方法同样只在训练折上 fit，测试折只 transform 和 predict。

### CNN

方法名：

```text
cnn
```

已在 `src/ultrasound_decoding/deep.py` 中实现 PyTorch backend。输入是原始 `128 x 501` 数据，经过 `arcsinh` 变换和训练集标准化后送入小型 2D CNN。

当前 `.venv` 中已经可以导入 PyTorch。深度模型会按每个 CV fold 重新训练，因此比线性模型慢很多。若只想快速复现实验，建议先显式指定 `--methods pca_lda cpca_lda`。

### fCNN

方法名：

```text
fcnn
```

这是单 session 解码用的极简全连接基线。输入为单帧 `[B, 1, 128, 501]`，先经过 `2 x 2` max pooling，下采样为约 `[B, 1, 64, 250]`，随后 `Flatten -> Linear(64 x 250, 3) -> ReLU -> Linear(3, n_classes)` 分类。该模型和其他 PyTorch 模型共用现有训练流程：每个外层 fold 只用训练 fold 统计做 `arcsinh` + pixel-wise z-score 归一化，并继续按 `cycle` 分组交叉验证。

### fCNN paper 32

方法名：

```text
fcnn_paper_32
```

这是论文结构的严格轻量复现版本。输入为单帧 `[B, 1, 128, 501]`，包含 4 个 `3 x 3` 卷积层，每层 32 个 filters，默认 ELU，前三层后接 `MaxPool2d(2)`，最后使用 global average pooling 和 `Linear(32, 2)` 分类。模型保留 `last_conv`，`forward(..., return_features=True)` 可返回 logits 和最后一层卷积特征图，便于 XGrad-CAM。

默认训练配置：

```text
optimizer = Adam
lr = 1e-3
weight_decay = 0
batch_size = 32
max_epochs = 80
activation = ELU
loss = CrossEntropyLoss
```

### fUS Lite CNN

方法名：

```text
fus_lite_cnn
```

这是针对当前小样本二维 fUS 单帧数据的轻量适配 CNN。输入为 `[B, 1, 128, 501]`，通道数为 `16 -> 32 -> 64 -> 64`，默认使用 GroupNorm + ELU；前三个卷积模块后下采样，最后一层不再池化，以保留更高的 CAM 空间分辨率。分类头为 global average pooling、dropout 和 `Linear(64, 2)`。

默认训练配置：

```text
optimizer = AdamW
lr = 1e-3
weight_decay = 1e-4
batch_size = 16
max_epochs = 80
early_stopping_patience = 12
activation = ELU
normalization = GroupNorm
dropout = 0.3
loss = CrossEntropyLoss
```

深度模型训练时每个外层 cycle-wise CV fold 都会重新初始化模型。新增 CNN 默认在训练 fold 内部按 cycle 留出 validation cycle，用 validation balanced accuracy 选择 epoch；选出 epoch 后会从头初始化，并使用完整 outer train cycles 重新训练到该 epoch，再评估 outer test fold。这样 outer test fold 不参与 epoch 选择，同时最终模型不会少用内部 validation cycle。若训练 fold 的 cycle 数不足以再按组划分验证集，会使用固定 epoch 在完整 outer train 上训练，不使用外层测试 fold 选 epoch。

### CNN + LSTM

方法名：

```text
cnn_lstm
```

已在 `src/ultrasound_decoding/deep.py` 中实现 PyTorch backend。模型先用 1D CNN 沿时间维提取特征，再用 LSTM 汇总时序信息，最后分类。

该模型也会按 fold 重新训练，运行时间通常长于普通 CNN。

## 代码结构

新增代码主要在：

```text
src/ultrasound_decoding/
  __init__.py
  io.py
  labels.py
  datasets.py
  cv.py
  evaluate.py
  linear.py
  deep.py

scripts/
  run_single_session_decoding.py
  run_model_batch_test.py
  run_cnn_architecture_benchmark.py
```

各文件职责：

| 文件 | 作用 |
| --- | --- |
| `io.py` | 读取 MATLAB v7.3/HDF5 的 `Data_SVD`，按数字顺序排序文件 |
| `labels.py` | 根据 120 秒周期和 30 秒 block 推断标签，并筛选 block 中间帧 |
| `datasets.py` | 组织单 session 数据，返回 `X, y, groups, metadata` |
| `cv.py` | 按 cycle 分组做交叉验证 |
| `evaluate.py` | 计算 accuracy、balanced accuracy、macro-F1、混淆矩阵 |
| `linear.py` | NumPy 实现 PCA、cPCA、LDA |
| `deep.py` | PyTorch 实现 CNN、CNN+LSTM、`fcnn`、`fcnn_paper_32`、`fus_lite_cnn` 和模型注册 |
| `run_single_session_decoding.py` | 统一运行入口和结果输出 |
| `run_model_batch_test.py` | 调用单 session 入口，批量测试 `pca_lda`、`cpca_lda`、`cnn`、`fcnn` |
| `run_cnn_architecture_benchmark.py` | 单帧轻量 2D-CNN 架构 benchmark 入口 |

## 运行方式

推荐先运行 708 的二分类：

```bash
.venv/bin/python scripts/run_single_session_decoding.py \
  --session 708 \
  --task binary
```

只跑线性模型：

```bash
.venv/bin/python scripts/run_single_session_decoding.py \
  --session 708 \
  --task binary \
  --methods pca_lda cpca_lda
```

运行新增 CNN，并使用同一组随机种子：

```bash
.venv/bin/python scripts/run_single_session_decoding.py \
  --session 708 \
  --task binary \
  --model fcnn_paper_32 fus_lite_cnn \
  --seeds 0 1 2 \
  --device auto
```

覆盖深度模型训练配置：

```bash
.venv/bin/python scripts/run_single_session_decoding.py \
  --session 708 \
  --task stimulus_type \
  --model fus_lite_cnn \
  --batch-size 8 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --max-epochs 80 \
  --patience 12
```

输出默认不会覆盖已有同名结果。如果目标 stem 已存在，runner 会自动追加形如 `run_YYYYMMDD_HHMMSS_xxxxxx` 的 run id；也可以手动指定：

```bash
.venv/bin/python scripts/run_single_session_decoding.py \
  --session 708 \
  --task binary \
  --model fcnn_paper_32 \
  --run-id cnn_arch_seed0
```

只有显式传入 `--overwrite` 时，才会覆盖同 stem 的已有输出。

运行两种刺激类型解码：

```bash
.venv/bin/python scripts/run_single_session_decoding.py \
  --session 708 \
  --task stimulus_type \
  --methods pca_lda cpca_lda
```

不丢弃 block 边界帧：

```bash
.venv/bin/python scripts/run_single_session_decoding.py \
  --session 708 \
  --task binary \
  --no-clean-middle
```

修改 block 边界 margin：

```bash
.venv/bin/python scripts/run_single_session_decoding.py \
  --session 708 \
  --task binary \
  --clean-margin-s 6
```

指定自定义文件编号范围：

```bash
.venv/bin/python scripts/run_single_session_decoding.py \
  --session 709 \
  --task binary \
  --analysis-limit 5:305
```

## 输出结果

每次运行会输出到：

```text
reports/decoding/
```

文件命名格式：

```text
summary/{session}_{task}_summary.json
metrics/{session}_{task}_overall_metrics.csv
metrics/{session}_{task}_fold_metrics.csv
metrics/{session}_{task}_confusion_matrix.csv
audit/{session}_{task}_cycle_selection_report.csv
audit/{session}_{task}_normalization_stats.csv
history/{session}_{task}_training_history.csv
permutation/{session}_{task}_{method}_seed{seed}_permutation_distribution.csv
permutation/{session}_{task}_{method}_seed{seed}_permutation_pvalues.csv
predictions/{session}_{task}_predictions.csv
samples/{session}_{task}_samples.csv
models/{session}_{task}_{method}_seed{seed}_fold{fold}_best.pt
audit/complete_cycle_audit.csv
```

例如 708 二分类：

```text
reports/decoding/summary/708_binary_summary.json
reports/decoding/metrics/708_binary_overall_metrics.csv
reports/decoding/metrics/708_binary_fold_metrics.csv
reports/decoding/metrics/708_binary_confusion_matrix.csv
reports/decoding/audit/708_binary_cycle_selection_report.csv
reports/decoding/permutation/708_binary_pca_lda_permutation_distribution.csv
reports/decoding/permutation/708_binary_pca_lda_permutation_pvalues.csv
reports/decoding/predictions/708_binary_predictions.csv
reports/decoding/samples/708_binary_samples.csv
```

各输出含义：

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 本次运行配置、样本数、类别数、每类样本数、CV 设置、整体指标、混淆矩阵、skipped 模型 |
| `overall_metrics.csv` | 每个方法跨全部测试折拼接后的 accuracy、balanced accuracy、macro-F1 |
| `fold_metrics.csv` | 每个方法每一折的指标 |
| `confusion_matrix.csv` | 每个方法的混淆矩阵长表 |
| `cycle_selection_report.csv` | 当前 session 在 analysis limit 后每个 cycle 的帧数、是否完整、首尾 index |
| `normalization_stats.csv` | CNN 每个 fold 的训练折归一化配置、训练/测试 NaN、Inf、全零图像检查和统计摘要 |
| `training_history.csv` | CNN 每个 fold 的 epoch 级训练记录；包含内部 epoch selection 和完整 outer train retrain 两个阶段 |
| `{method}_seed{seed}_permutation_distribution.csv` | 指定模型和随机种子的 label permutation test 每次置换 null 指标 |
| `{method}_seed{seed}_permutation_pvalues.csv` | 指定模型和随机种子的观测指标、null 均值/方差/范围、one-sided permutation p 值 |
| `predictions.csv` | 每个测试样本的真实标签、预测标签、所属 fold、cycle、block |
| `samples.csv` | 实际参与解码的样本元数据 |
| `models/*_best.pt` | CNN 每个 fold 根据 validation balanced accuracy 选择的最佳 epoch checkpoint |
| `complete_cycle_audit.csv` | `data/` 下所有 session 的完整周期裁剪总览 |

## Model Batch Test

批量测试 7 个 session、二分类和刺激类型两个任务，并比较：

```text
pca_lda
cpca_lda
cnn
fcnn
```

运行：

```bash
.venv/bin/python scripts/run_model_batch_test.py \
  --seeds 0 1 2 \
  --device auto
```

该脚本只负责批量编排，会逐个调用现有 `run_single_session_decoding.py`，因此不改变单 session 解码流程，并继续沿用 cycle-wise CV、训练 fold 归一化和每个 fold 重新初始化模型。默认发现 `data/` 下所有数字 session；当前为 708、709、710、807、813、817、822。输出默认写到：

```text
results/model_batch_test/
```

除了单 session 入口原有的 `summary/`、`metrics/`、`predictions/`、`audit/`、`history/`、`models/` 等结构化输出，还会额外生成：

```text
summary.csv
fold_metrics_all.csv
predictions_all.csv
batch_test_config.json
```

## CNN 架构 benchmark

新增脚本：

```bash
.venv/bin/python scripts/run_cnn_architecture_benchmark.py \
  --sessions 708 \
  --tasks binary stimulus_type \
  --seeds 0 1 2 \
  --device auto
```

第一阶段默认只训练并比较单帧输入的：

```text
fcnn
fcnn_paper_32
fus_lite_cnn
```

输出默认写到：

```text
results/cnn_architecture_benchmark/
```

脚本会使用现有 `run_single_session_decoding.py`，因此沿用相同样本选择、cycle-wise CV、训练 fold 归一化和 fold 内模型重初始化。若 `reports/decoding/metrics/` 中已有 `pca_lda`、`cpca_lda`、`cnn` 结果，脚本会把这些已有结果读入 `summary.csv` 方便横向查看，但不会重新实现或重跑传统方法。

benchmark 输出同样默认防覆盖：若 `results/cnn_architecture_benchmark/` 已有内容，会自动创建新的 `run_.../` 子目录；传 `--run-id` 可指定子目录名，传 `--overwrite` 才覆盖。

## Label Permutation Test

runner 支持 label permutation test。默认不运行；通过 `--n-permutations` 开启。

实现方式：

1. 保持原始样本、特征和按 cycle 的 CV split 不变。
2. 每次 permutation 全局打乱标签 `y`，保留总体类别数量。
3. 用打乱后的训练标签训练模型。
4. 用打乱后的测试标签计分。
5. 重复得到 null distribution。
6. 用真实标签下的观测分数和 null distribution 比较。

p 值使用 one-sided greater-or-equal 形式：

```text
p = (1 + number(null_score >= observed_score)) / (n_permutations + 1)
```

推荐先在线性模型上跑 permutation test：

```bash
.venv/bin/python scripts/run_single_session_decoding.py \
  --session 708 \
  --task binary \
  --methods pca_lda cpca_lda \
  --n-permutations 100 \
  --permutation-metric macro_f1 \
  --seed 0
```

输出：

```text
reports/decoding/permutation/708_binary_pca_lda_permutation_distribution.csv
reports/decoding/permutation/708_binary_pca_lda_permutation_pvalues.csv
reports/decoding/permutation/708_binary_cpca_lda_permutation_distribution.csv
reports/decoding/permutation/708_binary_cpca_lda_permutation_pvalues.csv
```

说明：

- permutation test 会对每个方法、每次 permutation 重新跑完整 grouped CV。
- permutation 输出按模型拆分命名；后续如果单独测试某个模型，可以直接从文件名看出模型名。
- 对 CNN/CNN+LSTM 开启 permutation test 会非常慢，建议先只用于线性模型。
- 如果要发表或做严谨统计，建议至少使用 1000 次 permutation；开发阶段可以先用 10 或 100 次快速检查。

## 当前验证结果

当前环境中 PyTorch 版本为 `2.8.0`。深度模型代码已经实现；为了快速生成稳定报告，本 README 中列出的正式结果来自 `pca_lda` 和 `cpca_lda`。一次默认命令曾完整跑完 `cnn` 的 708 二分类，随后在 `cnn_lstm` 训练时中断，因此没有把该次深度模型结果写入正式 CSV。

### 708 有无刺激二分类

命令：

```bash
.venv/bin/python scripts/run_single_session_decoding.py \
  --session 708 \
  --task binary
```

样本：

- `n_samples = 96`
- `n_cycles = 6`
- `no_stimulus = 48`
- `stimulus = 48`

整体结果：

| 方法 | Accuracy | Balanced Accuracy | Macro-F1 | 混淆矩阵类别顺序 |
| --- | ---: | ---: | ---: | --- |
| `pca_lda` | 0.792 | 0.792 | 0.791 | `no_stimulus`, `stimulus` |
| `cpca_lda` | 0.740 | 0.740 | 0.738 | `no_stimulus`, `stimulus` |

混淆矩阵：

```text
pca_lda:
[[40,  8],
 [12, 36]]

cpca_lda:
[[39,  9],
 [16, 32]]
```

100 次 label permutation test，使用 `macro_f1` 作为检验指标：

| 方法 | Observed Macro-F1 | Null Mean | Null Std | Null Min | Null Max | p value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pca_lda` | 0.791 | 0.485 | 0.064 | 0.322 | 0.635 | 0.0099 |
| `cpca_lda` | 0.738 | 0.486 | 0.064 | 0.331 | 0.646 | 0.0099 |

这里的 p 值是 one-sided greater-or-equal permutation p value。因为 100 次置换中没有任何一次 null score 超过观测 score，所以 p 值达到 100 次置换下的最小可分辨值 `1 / (100 + 1) = 0.0099`。

### 708 两种刺激类型二分类

命令：

```bash
.venv/bin/python scripts/run_single_session_decoding.py \
  --session 708 \
  --task stimulus_type \
  --methods pca_lda cpca_lda
```

样本：

- `n_samples = 48`
- `n_cycles = 6`
- `dot = 24`
- `grating = 24`

整体结果：

| 方法 | Accuracy | Balanced Accuracy | Macro-F1 | 混淆矩阵类别顺序 |
| --- | ---: | ---: | ---: | --- |
| `pca_lda` | 0.542 | 0.542 | 0.541 | `dot`, `grating` |
| `cpca_lda` | 0.604 | 0.604 | 0.596 | `dot`, `grating` |

混淆矩阵：

```text
pca_lda:
[[14, 10],
 [12, 12]]

cpca_lda:
[[18,  6],
 [13, 11]]
```

## 已做的关键工作

1. 梳理了项目已有数据和说明文档，确认 `readme.pptx` 中的视觉刺激范式。
2. 建立了单 session 数据加载接口，从 `.mat` 文件读取 `Data_SVD`。
3. 实现了基于文件编号、4 秒组间隔、120 秒周期的标签推断。
4. 实现了默认只取每个 30 秒 block 中间帧的清洁样本筛选策略。
5. 实现了按 cycle 分组的交叉验证，避免随机拆帧造成时间泄漏。
6. 用 NumPy 实现了 `pca_lda`，并保证 PCA 和 LDA 都只在训练折 fit。
7. 用 NumPy 实现了 `cpca_lda`，作为类别对比 PCA 加 LDA 的 baseline。
8. 用 PyTorch 代码实现了 `cnn` 和 `cnn_lstm`，当前环境 PyTorch 可用，但深度模型按 fold 训练耗时较长。
9. 实现了 accuracy、balanced accuracy、macro-F1、混淆矩阵等评估输出。
10. 增加了统一 runner，把 summary、metrics、audit、permutation、predictions、samples 分目录写入 `reports/decoding/`。

## 后续建议

下一步可以做三件事：

1. 为 `cnn` 和 `cnn_lstm` 增加更快的训练配置、早停和模型 checkpoint，再系统跑完整深度模型结果。
2. 对 709 使用 `--analysis-limit 5:305` 跑 early/high-response subset。
3. 增加跨 session 实验，例如 train on 708/710，test on 709。
