# SPATNIC

**S**patial **Pa**thology-based **T**ypist for **N**ormal / **C**ancer

A pip-installable tool for annotating tumor cells in single-cell and spatial transcriptomics data using a Student-t VAE classifier. Pre-trained models are specialized for **colorectal cancer (CRC)** tissue.

## Installation

```bash
pip install spatnic
```

## Quick Start

```python
import scanpy as sc
import spatnic

# Load and preprocess your data
adata = sc.read_h5ad("your_data.h5ad")
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# Predict (default: Tumor vs Normal)
spatnic.predict(adata)

# Results are added to adata.obs
print(adata.obs["spatnic_pred"].value_counts())
print(adata.obs["spatnic_score"].describe())        # Tumor probability
print(adata.obs["spatnic_score_Normal"].describe())  # Normal probability
print(adata.obs["spatnic_score_Tumor"].describe())   # Tumor probability
```

## Models

| Model | Description | Classes |
|-------|-------------|---------|
| `tumor_normal` (default) | Tumor vs normal epithelial cells | Normal, Tumor |
| `tumor_other` | Tumor vs all other cell types | Other, Tumor |

```python
# Select a model
spatnic.predict(adata, model="tumor_normal")
spatnic.predict(adata, model="tumor_other")

# Use a custom (fine-tuned) model
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

# Predict with the fine-tuned model
spatnic.predict(adata_new, model="my_finetuned.pt")
```

## Options

```python
spatnic.predict(
    adata,
    model="tumor_normal",      # model name or path
    threshold=0.5,              # classification threshold
    batch_size=2048,            # batch size for inference
    key_added="spatnic_pred",   # obs key for predicted labels
    score_key_added="spatnic_score",  # also adds spatnic_score_Normal, spatnic_score_Tumor
    layer_key=None,             # AnnData layer to use (None = .X)
    return_latent=False,        # if True, also returns latent representation
    device=None,                # "cuda" or "cpu" (auto-detected)
)
```

## Checkpoint Export (for developers)

After training in a notebook, export a checkpoint:

```python
from spatnic import export_checkpoint

export_checkpoint(
    state_dict_path="student_t_vae_cls_weights.pth",
    gene_names=list(adata_hvg.var_names),  # 3000 genes
    mu_g=mu_g,  # per-gene mean from training set
    std_g=std_g,  # per-gene std from training set
    model_name="tumor_normal",
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

## License

MIT
