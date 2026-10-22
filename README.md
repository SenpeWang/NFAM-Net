# NFAM-Net

**English** | [简体中文](README_zh-CN.md)

NFAM-Net jointly learns an explicit nematic field and semantic features for elongated-structure segmentation. Opposite tangent directions represent the same local axis; semantic features progressively refine the image-derived field, and the refined director and nematic order guide feature learning. This README covers three public datasets: NeuriteOutgrowth, NeuroQuantify, and ARCADE.

## Overview

The image-derived field is an analytical initialization. **DFR subsequently learns to refine that field** using semantic features; the initialization and the learned updates have different roles.

| Component | Role |
| --- | --- |
| **DFR — Director Field Refinement** | Updates the two field components through gated residual refinement before encoder blocks. |
| **DFE — Director Field Enhancement** | Projects shallow feature gradients parallel and perpendicular to the refined director, then encodes and fuses the responses. |
| **NOFM — Nematic Order-aware Feature Modulation** | Combines Soft-NIBS channel reweighting, SS2D context modeling, and an N-GLU conditioned on the refined order map. |
| **Decoder** | Integrates multi-scale encoder features through a UPerNet-style head to predict the segmentation mask. |

[![NFAM-Net framework](docs/figures/framework.png)](docs/figures/framework.pdf)

*NFAM-Net framework. Click the figure to open the PDF.*

The field has two components,

$$
q_1=S\cos(2\theta),\qquad q_2=S\sin(2\theta),\qquad S=\sqrt{q_1^2+q_2^2}.
$$

The double-angle encoding preserves head–tail symmetry. The network refines `q1` and `q2`; `S` is derived from them, rather than predicted as a third independent component. Raw nematic order describes local axial alignment, while the refined order map serves as a learned geometric cue. The raw order map supplies CR training supervision; the refined order map conditions the N-GLU forward gate. No additional orientation annotations are required.

### Module diagrams

The diagrams below are the current framework and module figures, exported as PNG for browser viewing with the original PDFs available for inspection.

<details>
<summary><strong>DFR — Director Field Refinement</strong></summary>

Semantic features predict a candidate two-component field; a shared spatial gate controls its residual update.

[![Director Field Refinement](docs/figures/dfr.png)](docs/figures/dfr.pdf)

[Open the DFR PDF](docs/figures/dfr.pdf)

</details>

<details>
<summary><strong>DFE — Director Field Enhancement</strong></summary>

Nematic projection separates gradients parallel and perpendicular to the director. The two branches are encoded and fused with a residual connection.

[![Director Field Enhancement](docs/figures/dfe.png)](docs/figures/dfe.pdf)

[Open the DFE PDF](docs/figures/dfe.pdf)

</details>

<details>
<summary><strong>NOFM — Nematic Order-aware Feature Modulation</strong></summary>

Soft-NIBS reweights channels, SS2D models long-range context, and the refined order map conditions N-GLU. Dashed paths in the diagram indicate CR training dependencies.

[![Nematic Order-aware Feature Modulation](docs/figures/nofm.png)](docs/figures/nofm.pdf)

[Open the NOFM PDF](docs/figures/nofm.pdf)

</details>

## Environment setup

Use Linux with an NVIDIA GPU and a CUDA-enabled PyTorch installation. The SS2D backend uses the bundled Mamba-3 **Triton SISO** implementation in [networks/VMamba](networks/VMamba); the supplied training and testing entry points call CUDA directly.

```bash
git clone https://github.com/SenpeWang/NFAM-Net.git
cd NFAM-Net
conda create -n nfam-net python=3.10 -y
conda activate nfam-net
python -m pip install -r requirements.txt
```

Install a PyTorch build appropriate for your GPU and CUDA environment using the [official installation instructions](https://pytorch.org/get-started/locally/), then check the environment:

```bash
python -c "import torch, triton; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'Triton:', triton.__version__); assert torch.cuda.is_available()"
python train.py --help
python test.py --help
```

The inspected development environment uses Python **3.10.20**, PyTorch **2.12.0+cu130**, Triton **3.7.0**, and NumPy **2.2.6**. The lower bounds in [requirements.txt](requirements.txt) are dependency specifications, not a tested compatibility matrix. A separate `mamba-ssm` or `selective_scan` extension installation is not used by this checkout's default backend.

## Data preparation

Set each dataset's `root` in [datasets/dataset_config.py](datasets/dataset_config.py) to its location on your machine. The paths currently in that file are local development paths.

| Dataset argument | Input channels | Array layout | Mask directory |
| --- | ---: | --- | --- |
| `NeuriteOutgrowth` | 1 | `HW` | `labels/` |
| `NeuroQuantify` | 1 | `HW` | `labels/` |
| `ARCADE` | 1 | `HW` | `masks/` |

For NeuriteOutgrowth and NeuroQuantify, arrange each root as follows:

```text
DATASET_ROOT/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

For ARCADE, use **1,000 training images and 200 test images**:

```text
ARCADE_ROOT/
├── train/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

The ARCADE registry maps the logical `val` split to `train`. The checkpoint-selection pass therefore uses the training split; `test.py` evaluates the separate test split. A `val/` directory is not required for this configuration.

Images may be `.tif`, `.tiff`, or `.png`. Each image must have a same-stem `.png` mask, such as `images/case001.tif` and `labels/case001.png` (or `masks/case001.png` for ARCADE). Nonzero mask pixels are foreground. The loader resizes image–mask pairs to **512 × 512** by default. Single-channel public datasets use the normalization and augmentation settings in the registry.

### Public dataset sources

- **NeuriteOutgrowth:** [author repository and dataset link](https://github.com/YoavLotem/NeuriteOutgrowth).
- **NeuroQuantify:** [Edmond dataset, DOI: 10.17617/3.UIBMJX](https://doi.org/10.17617/3.UIBMJX).
- **ARCADE:** [dataset repository](https://github.com/cmctec/ARCADE).

Download and organize the data before running the entry points. Dataset files and model checkpoints are not bundled with this repository.

## Training

After configuring the dataset roots, train an ARCADE model:

```bash
python train.py --dataset ARCADE --batch_size 2 --lr 3e-4 --seed 1234 --gpu 0
```

The same entry point supports the other two public datasets:

```bash
python train.py --dataset NeuriteOutgrowth --gpu 0
python train.py --dataset NeuroQuantify --gpu 0
```

`--gpu` selects the visible GPU index. Run directory names are generated automatically under `Outputs/Train/`. Use `--tag` to label an experiment or `--output_root` to store it elsewhere.

### Defaults and configuration

These are the current **training entry-point defaults**, not a claim that one setting is optimal for every dataset.

| Setting | Default | Where to change it |
| --- | --- | --- |
| Batch size | 2 | `--batch_size` |
| Learning rate | `3e-4` | `--lr` |
| Random seed | 1234 | `--seed` |
| EMA | Enabled | `--ema` / `--no-ema` |
| Soft-NIBS channel-weight lower bound | 0.70 | `--nibs_floor` |
| Nematic field loss weight | 0.30 | `--lambda_nematic` |
| CR information / bottleneck weights | 0.03 / 0.05 | `--lambda_cr_info` / `--lambda_cr_b` |
| Order / Frank–Oseen weights | 1.0 / 1.0 | `--s_weight` / `--fo_weight` |
| BCE / Dice weights | 0.3 / 0.7 | `--bce_weight` / `--dice_weight` |

Additional settings are defined in [configs/config.py](configs/config.py). In particular, `train.py` overrides some loss weights from `Config`; inspect the saved `config.yaml` for the settings actually used by a run.

## Testing and visualization

Set `RUN_DIR` to an actual training directory containing `checkpoints/`:

```bash
RUN_DIR="Outputs/Train/your_run"
python test.py --dataset ARCADE --train_dir "$RUN_DIR" --ckpt_type best_dice --gpu 0
```

By default, testing saves binary prediction masks and metric reports. Add `--vis` to also save image/GT/prediction comparisons and stage-wise nematic-field visualizations:

```bash
python test.py --dataset ARCADE --train_dir "$RUN_DIR" --ckpt_type best_dice --vis --gpu 0
```

For metrics without saved images:

```bash
python test.py --dataset ARCADE --train_dir "$RUN_DIR" --ckpt_type best_dice --no_save_images --gpu 0
```

You can also provide a checkpoint directly:

```bash
python test.py --dataset ARCADE --ckpt "$RUN_DIR/checkpoints/best_dice_model.pth" --gpu 0
```

Supported checkpoint selectors are `best_dice`, `best_loss`, `best_ema`, and `last`; the default is `best_dice`. An EMA checkpoint exists only if EMA was enabled and an improvement was saved. Use the dataset, architecture, and preprocessing settings corresponding to the checkpoint. For ablation weights, the mode is inferred from the run's `summary.json` or directory name, and can be set explicitly with `--ablation`.

The evaluator thresholds the foreground softmax probability at **0.5** and reports **Dice, IoU, Precision, Recall, and clDice** on a 0–1 scale. Means and standard deviations are computed across valid images. Images for which both the mask and prediction are empty are excluded from these averages; their `valid` flag is retained in `test_report.json`.

## Component ablations

```bash
python train.py --dataset ARCADE --ablation baseline --gpu 0
python train.py --dataset ARCADE --ablation wo_dfr --gpu 0
```

| `--ablation` | Change from the full model |
| --- | --- |
| `full` | All nematic components enabled. |
| `baseline` | Removes the nematic prior, field updates, directional projection, Soft-NIBS, and order conditioning; retains the convolutional/state-space backbone. |
| `wo_dfe` | Bypasses nematic projection in the shallow blocks. |
| `wo_dfr` | Removes learned field updates while retaining feature learning and the image-derived field. |
| `wo_nibs` | Bypasses Soft-NIBS reweighting and disables its CR losses. |
| `wo_sgate` | Removes refined-order conditioning from the deep gate. |

Loss and channel-floor studies use the corresponding weight arguments in the defaults table. Soft-NIBS is a **channel-reweighting mechanism**, not a topological boundary constraint.

## Representation experiments

The additional scripts compare signed polar fields (`P`) with sign-augmented polar fields (`P_AUG`) on ARCADE:

```bash
python scripts/train_arcade_polar.py --group P --dataset ARCADE --gpu 0
python scripts/train_arcade_polar.py --group P_AUG --dataset ARCADE --gpu 0
```

Set `POLAR_RUN_DIR` to the matching run and select the group used to train it:

```bash
POLAR_RUN_DIR="Outputs/Polar/Train/your_run"
python scripts/eval_arcade_polar.py --group P_AUG --dataset ARCADE --train_dir "$POLAR_RUN_DIR" --ckpt_type best_dice --gpu 0
```

This evaluates the fixed checkpoint under `normal`, `global_flip`, `alternate_halfplane`, and `quadrants` direction assignments. The results are saved as `eval_best_dice_reversal_results.json` inside the run directory. Sign augmentation is an experimental comparison; it does not make a polar representation structurally head–tail symmetric.

[scripts/eval_arcade_axis_prior_study.py](scripts/eval_arcade_axis_prior_study.py) separately analyzes **geometric representation methods** on the ARCADE test split. Set `img_dir`, `mask_dir`, and `out_dir` in its `main()` before running it. Its Markdown export additionally requires `tabulate` (`python -m pip install tabulate`). This utility's test-split analysis is distinct from a full-cohort analysis across all datasets.

## Outputs

```text
Outputs/
├── Train/<run>/
│   ├── config.yaml
│   ├── summary.json
│   ├── metrics.txt
│   ├── loss_details.txt
│   ├── tensorboard/
│   └── checkpoints/
│       ├── best_dice_model.pth
│       ├── best_loss_model.pth
│       ├── best_ema_model.pth
│       ├── last_model.pth
│       └── last_training_state.pt
└── Test/<run>/
    ├── test_report.json
    ├── test_report.txt
    ├── pred/
    ├── comparison/             # with --vis
    └── nematic/stage0...stage3/ # with --vis
```

`pred/` is omitted with `--no_save_images`. TensorBoard can read the training logs with `tensorboard --logdir Outputs/Train`.

## Repository layout

```text
configs/     Training and model configuration
datasets/    Dataset registry, loading, and augmentation
networks/    NFAM-Net, DFR, DFE, NOFM, Q-tensor operations, and decoder
  VMamba/    SS2D and the bundled Mamba-3 backends
utils/       Losses, metrics, visualization, and logging
experiment/  Polar-field representation and loss variants
scripts/     ARCADE representation and geometric-analysis experiments
train.py     Training entry point
test.py      Evaluation entry point
```

## Using a new dataset

1. Arrange the image/mask pairs in the split directories shown above.
2. Add an entry to `DATASET_REGISTRY` with the root, channel count, array layout, and mask directory. Optional fields include `label_suffix`, `normalize`, `intensity_aug`, `oversample`, and `split_map`.
3. Add the dataset name to the `--dataset` choices in both `train.py` and `test.py`.
4. Train and evaluate through the same entry points, using a matching data configuration.

## Acknowledgments

The state-space pathway builds on VMamba and the bundled Mamba-3 Triton implementation. The Mamba-3 kernel files retain their Dao AI Lab / Goombalab copyright notices. See the corresponding source files for attribution.
