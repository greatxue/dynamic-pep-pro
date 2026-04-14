# Bug Fixes (2026-04-14)

## 1. `validation_step` 单 dataloader 时崩溃

**文件**: `src/proteinfoundation/proteina.py`

**问题**: `validation_step(self, batch, batch_idx, dataloader_idx: int)` 没有默认值。PepMerge 配置只有一个 val dataloader（无 `generation` 配置），Lightning 要求单 dataloader 时参数必须有默认值，否则抛出 `RuntimeError`。

**错误**:
```
RuntimeError: You provided only a single `val_dataloader`, but have included
`dataloader_idx` in `Proteina.validation_step()`.
```

**修复**: 加了 `dataloader_idx=0` 默认值，并在运行时检测 val dataloader 数量，单个 dataloader 时直接走 `validation_step_data`，避免错误路由到 `validation_step_lens`（生成逻辑）。

---

## 2. Val loss 未被 log

**文件**: `src/proteinfoundation/proteina.py`

**问题**: `on_validation_epoch_end_data` 只是清空了 `self.validation_output_data` 列表，从未调用 `self.log()`，导致 val loss 收集了但不显示在 wandb。

**修复**: 在清空前计算均值并调用 `self.log("val/loss", avg_val_loss, sync_dist=True)`。

---

## 3. Train/val 分割未 shuffle

**文件**: `src/proteinfoundation/datasets/structure_data.py`

**问题**: `full_metadata.iloc[:n_train]` / `iloc[n_train:]` 按 CSV 行顺序切割，若 CSV 有任何排序规律（按质量、来源、长度等）会导致 train 和 val 分布不一致，人为放大 train/val loss gap。

**修复**: 分割前加 `full_metadata.sample(frac=1, random_state=42).reset_index(drop=True)`，固定随机种子保证可复现。

---

## 4. `FilterTargetResiduesTransform` 未更新 `num_nodes` —— 核心训练 Bug

**文件**: `src/proteinfoundation/datasets/transforms.py`

**问题**: `FilterTargetResiduesTransform` 将 target 残基从主特征中过滤掉后，`graph.coords_nm` 等张量变为 `[n_binder, ...]`，但 PyG 的 `graph.num_nodes` 仍是创建时存储的原始值 `n_total = n_binder + n_target`，未被更新。

Collation 中 `binder_lengths = [s.num_nodes]` 拿到的是 `n_total`（如 116），导致：
- `batch["mask"]` 大小为 116，全为 True
- `coords_nm` 实际只有 9 个残基（binder），padding 到 116 后位置 9-115 全是零
- Flow matching loss 在这 107 个零坐标位置上也计算梯度，目标速度为 `0 - noise`
- 模型被迫学习将 ~90% 的位置推向原点，训练信号完全错误

**影响**: 所有使用 `FilterTargetResiduesTransform` 的训练 run 均受影响，训练出的 checkpoint 无效，需废弃并重新训练。

**修复**: 在 `FilterTargetResiduesTransform` 末尾（清理 `target_residue_mask` 之后）显式更新：
```python
graph.num_nodes = int(binder_residue_mask.sum())
```

---

## 5. `CroppingTransform2` binder 链选择错误 —— 系统性选链 Bug

**文件**: `src/proteinfoundation/datasets/transforms.py`

**问题**: `CroppingTransform2` 用 argmin（最短 interface chain）启发式方法选 binder 链。PepMerge 的 complex.pdb 由肽链（chain A）+ 口袋（chain B）组成，但：
1. 有些 PDB 文件里 chain B 的 ATOM 记录先出现，导致 `chain_names` 里顺序不定
2. 口袋有时被拆成多段，某段恰好比肽链短，argmin 选中口袋段当 binder

结果：Pad binder 显示 40-68（口袋），而不是 10-25（肽链），模型学的是生成口袋而非肽链。

**修复**: 给 `CroppingTransform2` 加 `binder_chain_name: str | None = None` 参数。指定时通过 `graph.chain_names`（按 PDB 字母名查找，不依赖 ATOM 记录顺序）直接定位 binder 链，跳过 argmin 启发式。在 `pepmerge_dataset.yaml` 里设 `binder_chain_name: "A"`。

---

## 6. 多卡训练显存严重不均衡

**文件**: `src/proteinfoundation/train.py`

**问题**: DDP 训练时几乎所有显存集中在 rank 0（第一张卡），其他卡显存占用极低。通常原因是 loss 在 rank 0 上聚合计算，或 collation/数据预处理在主进程上完成后广播。

**修复**: 在 `train.py` 打补丁（具体改动待补充）。

---

## 修复后预期行为

- `Pad binder=` 日志应显示真实肽链长度（~10–25），而非 40-68 或 100+
- `groups={'target': ...}` 显示口袋/受体长度（~30–240）
- `val/loss` 指标正常出现在 wandb
- Flow matching 只在真实 binder 残基上计算 loss
- 多卡显存均衡，各卡占用接近
