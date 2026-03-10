# SPATNIC

**S**patial **Pa**thology-based **T**ypist for **N**ormal / **C**ancer

Single-cell/spatial transcriptomics data から腫瘍細胞をアノテーションするツール。

## Install

```bash
pip install /path/to/spatnic
```

## Quick Start

```python
import scanpy as sc
import spatnic

# データの読み込みと前処理
adata = sc.read_h5ad("your_data.h5ad")
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# 予測（デフォルト: Tumor vs Normal）
spatnic.predict(adata)

# 結果は adata.obs に追加される
print(adata.obs["spatnic_pred"].value_counts())
print(adata.obs["spatnic_score"].describe())
```

## Models

| Model | Description | Classes |
|-------|-------------|---------|
| `tumor_normal` (default) | がん細胞 vs 正常上皮細胞 | Normal, Tumor |
| `tumor_other` | がん細胞 vs その他の細胞 | Other, Tumor |

```python
# モデルを指定
spatnic.predict(adata, model="tumor_normal")
spatnic.predict(adata, model="tumor_other")

# カスタムモデル（fine-tuning後）
spatnic.predict(adata, model="path/to/finetuned.pt")
```

## Fine-tuning

```python
spatnic.finetune(
    adata,
    label_key="cell_type",
    label_map={"Epithelial": 0, "Malignant": 1},
    save_path="my_finetuned.pt",
)

# fine-tuned モデルで予測
spatnic.predict(adata_new, model="my_finetuned.pt")
```

## Options

```python
spatnic.predict(
    adata,
    model="tumor_normal",    # モデル名 or パス
    threshold=0.5,            # 分類閾値
    batch_size=2048,          # バッチサイズ
    key_added="spatnic_pred", # obs に追加するキー名
    score_key_added="spatnic_score",
    layer_key=None,           # 使用するlayer（Noneなら .X）
    return_latent=False,      # Trueなら潜在表現も返す
    device=None,              # "cuda" or "cpu"（自動検出）
)
```

## Checkpoint Export (for developers)

ノートブックで学習した後、チェックポイントをエクスポート:

```python
from spatnic import export_checkpoint

export_checkpoint(
    state_dict_path="student_t_vae_cls_weights_primary.pth",
    gene_names=list(adata_hvg.var_names),  # 3000 genes
    mu_g=mu_g,  # per-gene mean from training set
    std_g=std_g,  # per-gene std from training set
    save_path="tumor_normal.pt",
    label_map={0: "Normal", 1: "Tumor"},
)
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- scanpy >= 1.9
- anndata >= 0.8

## Performance (default model: tumor_normal)

```
              precision    recall  f1-score   support
      Normal       0.96      0.96      0.96     48902
       Tumor       0.96      0.96      0.96     48902
    accuracy                           0.96     97804
ROC-AUC: 0.9909
```
