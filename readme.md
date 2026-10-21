# NFAM-Net

**NFAM-Net: Nematic Field-Aware Modeling Network** — a segmentation network for directional,
elongated structures (neurites, vessels, fibers) that injects a liquid-crystal **nematic Q-tensor
physics prior** and lets the network co-evolve with it.

Instead of treating orientation as a learned latent variable, NFAM-Net extracts an explicit
nematic order-parameter field (scalar order parameter *S* + director angle *2θ*) directly from
the raw image with a training-free estimator, then keeps this physical field alive inside the
network: it is fused into the stem, decomposes shallow feature gradients onto director bases,
is dynamically refined by semantic features, gates the deep-stage GLU, and drives a
soft information-bottleneck channel selection.

## Architecture

```
Image ──► Stem (+Q prior concat) ──► DFE ×2 ──► NOFM ×2 ──► UPerNet head ──► Segmentation
              ▲                        ▲            ▲
              │ concat                 │ gated      │ channel gate
              │                     Q update      │
          QImagePrior ──► DFR (dynamic Q evolution) ──► NIBS
          (training-free)      Nematic Q-field path
```

| Module | Role |
| :--- | :--- |
| `QImagePrior` ([networks/qtensor.py](networks/qtensor.py)) | Training-free Q-tensor prior: gradient-energy-weighted structure tensor → order parameter *S* and director field (q1, q2) |
| `DFEBlock` / `NematicConv` ([networks/dfe.py](networks/dfe.py)) | Shallow directional feature extraction: decomposes feature gradients onto parallel/perpendicular director bases |
| `DFR` ([networks/dfr.py](networks/dfr.py)) | Dynamic field refinement: predicts an ideal Q state from semantics and applies a norm-capped, gate-controlled convex update |
| `NOFMBlock` / `NematicGLU` ([networks/nofm.py](networks/nofm.py)) | Deep stage block: 4-directional selective scan (Mamba3 SS2D) + N-GLU gated by the scalar order parameter *S* |
| `NIBS` ([networks/nibs.py](networks/nibs.py)) | Soft-bounded nematic information bottleneck: differentiable channel selection with a throughput floor |
| Losses ([utils/losses.py](utils/losses.py)) | *S* supervision (fg→1, bg→0) + Frank-Oseen elastic energy + CR information/bottleneck regularization |

Backbone: CNN stem + VMamba (Mamba3 SISO SS2D) hybrid encoder, UPerNet decoder.
~53M parameters; the nematic mechanisms themselves add only ~0.5M.

## Repository Layout

```
NFAM-Net/
├── readme.md / requirements.txt
├── train.py / test.py        # training and testing entry points
├── configs/                  # global configuration
├── datasets/                 # dataset registry and loader
├── networks/                 # model core (incl. vendored VMamba/mamba3 kernels)
├── utils/                    # losses, metrics, loggers, visualization
├── experiment/               # native polar-vector (P) representation study
└── scripts/                  # ARCADE experiments: polar pipeline + axis-prior study
```

## Installation

```bash
conda create -n nfam python=3.10 -y
conda activate nfam
pip install torch --index-url https://download.pytorch.org/whl/cu124   # or your CUDA build
pip install -r requirements.txt
```

## Dataset Setup

The public benchmark used in this project is **ARCADE** (coronary angiography segmentation,
[ARCADE challenge](https://arcade.grand-challenge.org/)): ~1000 training and 200 test images
with binary vessel masks.

Expected layout is `<root>/<split>/images` + `<root>/<split>/labels`
(ARCADE uses `masks` instead of `labels`). Dataset roots are registered in
[datasets/dataset_config.py](datasets/dataset_config.py).

To train on your own data, add an entry to `DATASET_REGISTRY` specifying the channel count,
shape format and optional flags: z-score normalization, nnU-Net-style intensity augmentation,
foreground oversampling, and split mapping.

## Training

```bash
# full model (200 epochs by default)
python train.py --dataset ARCADE --gpu 0

# ablation variants
python train.py --dataset ARCADE --ablation baseline --gpu 0   # pure Conv + SSM backbone
python train.py --dataset ARCADE --ablation wo_dfr --gpu 0     # static Q prior (no dynamic refinement)
```

Ablation modes: `full`, `baseline`, `wo_dfe`, `wo_dfr`, `wo_nibs`, `wo_sgate`.
Key options: `--epochs`, `--batch_size`, `--lr`, `--seed`, `--ema/--no-ema` (EMA on by default),
`--lambda_nematic`, `--fo_weight`, `--s_weight`, `--nibs_floor`, `--tag`, `--resume`.
Outputs land in `Outputs/Train/<timestamp>/` with checkpoints (`best_dice`, `best_loss`,
`best_ema`, `last`), TensorBoard logs, metrics and loss breakdowns.

## Testing

```bash
python test.py --dataset ARCADE --train_dir Outputs/Train/<run_dir> \
               --ckpt_type best_ema --gpu 0            # metrics only
python test.py --dataset ARCADE --train_dir Outputs/Train/<run_dir> --vis   # + comparison/nematic plots
```

`--ckpt_type` selects among `best_dice`, `best_loss`, `best_ema`, `last`.

## Polar-Vector Representation Study

[experiment/](experiment/) replaces the second-order Q-tensor with a first-order signed polar
vector **P** (native polarity, 180° sign ambiguity). Sign ambiguity is removed during training
by randomized sign augmentation (`P_AUG`), and [scripts/eval_arcade_polar.py](scripts/eval_arcade_polar.py)
quantifies residual sensitivity across four direction-reversal modes.
`forbid_q_runtime()` hard-fails any legacy Q-tensor call on the P path.

```bash
python scripts/train_arcade_polar.py --group P_AUG --dataset ARCADE --gpu 0
python scripts/eval_arcade_polar.py  --group P_AUG --dataset ARCADE --train_dir <run_dir>
```

## Notes

- The loader handles both multi-channel tiff stacks (CHW) and single-channel PNG images
  through the same registry.
- The nematic prior is training-free: `QImagePrior` runs under `torch.no_grad()` and adds no
  learned parameters.
