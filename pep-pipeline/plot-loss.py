import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# ========= 0️⃣ 读取 =========
csv_path = "/mlx_devbox/users/xuezhongkai/playground/dynamic-pep-pro/Proteina-Complexa/store/ft-pepmerge-0415-mambaout2_local/lightning_logs/version_0/metrics.csv"

df = pd.read_csv(csv_path)
df = df.dropna(how="all")

# 转数值（防止字符串问题）
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ========= 1️⃣ 拆分 train / val =========
train_df = df[df["train_loss_step"].notna()].copy()
train_df = train_df.sort_values("step")

# ⚠️ 防止没有 validation 列时报错
if "validation_loss/loss_epoch" in df.columns:
    val_df = df[df["validation_loss/loss_epoch"].notna()].copy()
    val_df = val_df.sort_values("step")
else:
    val_df = pd.DataFrame()

# val bb_ca = total - latent（如果存在）
if not val_df.empty and "validation_loss/loss_local_latents_epoch" in val_df.columns:
    val_df["val_bb_ca"] = (
        val_df["validation_loss/loss_epoch"]
        - val_df["validation_loss/loss_local_latents_epoch"]
    )

# ========= 2️⃣ 平滑 =========
def smooth(x, k=20):
    return x.rolling(k, min_periods=1).mean()

# ========= 3️⃣ 自适应 y 轴 =========
def set_adaptive_ylim(ax, y_list, margin=0.1):
    values = []
    for v in y_list:
        if v is not None and len(v) > 0:
            values.append(v.dropna().values)

    if len(values) == 0:
        return

    y = np.concatenate(values)
    ymin, ymax = y.min(), y.max()

    if ymin == ymax:
        pad = abs(ymin) * 0.1 + 1e-6
    else:
        pad = (ymax - ymin) * margin

    ax.set_ylim(ymin - pad, ymax + pad)

# ========= 4️⃣ 画图 =========
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
fig.suptitle("Fine-tuning Training Curves", fontsize=14, y=1.01)

c_train = "#1f77b4"
c_val   = "#d62728"

# --- Total Loss ---
ax = axes[0]
ax.plot(train_df["step"], train_df["train_loss_step"],
        alpha=0.15, color=c_train, linewidth=0.8)
ax.plot(train_df["step"], smooth(train_df["train_loss_step"]),
        color=c_train, linewidth=2, label="Train (smoothed)")

if not val_df.empty:
    ax.plot(val_df["step"], val_df["validation_loss/loss_epoch"],
            "o-", color=c_val, markersize=5, linewidth=1.5, label="Val (epoch)")

ax.set_ylabel("Total Loss")
ax.set_title("Total Loss")
ax.legend()
ax.grid(alpha=0.3)

set_adaptive_ylim(ax, [
    train_df["train_loss_step"],
    val_df["validation_loss/loss_epoch"] if not val_df.empty else None
])

# --- Backbone (CA) Loss ---
ax = axes[1]
ax.plot(train_df["step"], train_df["train/loss_bb_ca_step"],
        alpha=0.15, color=c_train, linewidth=0.8)
ax.plot(train_df["step"], smooth(train_df["train/loss_bb_ca_step"]),
        color=c_train, linewidth=2, label="Train (smoothed)")

if not val_df.empty and "val_bb_ca" in val_df.columns:
    ax.plot(val_df["step"], val_df["val_bb_ca"],
            "o-", color=c_val, markersize=5, linewidth=1.5, label="Val (epoch)")

ax.set_ylabel("BB-CA Loss")
ax.set_title("Backbone (CA) Loss")
ax.legend()
ax.grid(alpha=0.3)

set_adaptive_ylim(ax, [
    train_df["train/loss_bb_ca_step"],
    val_df["val_bb_ca"] if not val_df.empty and "val_bb_ca" in val_df.columns else None
])

# --- Latent Loss ---
ax = axes[2]
ax.plot(train_df["step"], train_df["train/loss_local_latents_step"],
        alpha=0.15, color=c_train, linewidth=0.8)
ax.plot(train_df["step"], smooth(train_df["train/loss_local_latents_step"]),
        color=c_train, linewidth=2, label="Train (smoothed)")

if not val_df.empty and "validation_loss/loss_local_latents_epoch" in val_df.columns:
    ax.plot(val_df["step"], val_df["validation_loss/loss_local_latents_epoch"],
            "o-", color=c_val, markersize=5, linewidth=1.5, label="Val (epoch)")

ax.set_ylabel("Latent Loss")
ax.set_title("Local Latents Loss")
ax.set_xlabel("Training Step")
ax.legend()
ax.grid(alpha=0.3)

set_adaptive_ylim(ax, [
    train_df["train/loss_local_latents_step"],
    val_df["validation_loss/loss_local_latents_epoch"] if not val_df.empty else None
])

plt.tight_layout()

# ========= 5️⃣ 保存 =========
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"/mlx_devbox/users/xuezhongkai/playground/dynamic-pep-pro/pep-pipeline/plot/plot_training_curves_{timestamp}.png"

plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")

plt.show()