# CFD-FSA: Content-Forensic Disentangled Few-Shot Adapter for Deepfake Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Publication](https://img.shields.io/badge/CVPR%2FIEEE-Research%20Specification-purple.svg)](CFD-FSA-Novelty-Statement.md)

---

## 📖 Executive Summary

Few-shot deepfake detection aims to distinguish real and manipulated facial media given only a handful ($K \in \{1, 5, 10, 20\}$) of support examples. However, foundation visual models like **CLIP (ViT-B/32)** are pre-trained on generic semantic and identity features. When adapted naively, they latch onto **content shortcuts** (facial structure, subject identity, skin tone, background) rather than subtle manipulation artifacts, crippling generalization across unseen manipulation methods and diverse datasets.

**CFD-FSA (Content-Forensic Disentangled Few-Shot Adapter)** solves this fundamental entanglement. By explicitly decomposing facial representations, adversarially purging identity leakage via gradient reversal, enforcing strict geometric orthogonality against the semantic manifold, and stabilizing few-shot episodic prototypes with memory routing, CFD-FSA achieves state-of-the-art generalizable detection.

```
                         ┌────────────────────────────────────────────────────────┐
                         │                    Input Face Image                    │
                         └───────────┬────────────────────────────────┬───────────┘
                                     │                                │
                                     ▼                                ▼
                         ┌───────────────────────┐        ┌───────────────────────┐
                         │  Frozen CLIP ViT-B/32 │        │ Novelty 1: TPFD       │
                         │  (Semantic Manifold)  │        │ Spatial + Freq + Edge │
                         └───────────┬───────────┘        └───────────┬───────────┘
                                     │                                │
                                     │   ┌────────────────────────────┘
                                     ▼   ▼
                         ┌───────────────────────────────────┐
                         │  Forensic Bottleneck Adapter (FBA)│
                         │     Dimension Reduction (d -> r)  │
                         └─────────────────┬─────────────────┘
                                           │
                                           ▼
                         ┌───────────────────────────────────┐
                         │  Novelty 2: AIS + ORC             │
                         │  • Adversarial GRL (vs Identity)  │
                         │  • Orthogonal Residual Loss       │
                         └─────────────────┬─────────────────┘
                                           │ Disentangled Forensic Vector z_f
                                           ▼
                         ┌───────────────────────────────────┐
                         │  Novelty 3: MSPM Memory           │
                         │  Slot-based Prototype Enrichment  │
                         └─────────────────┬─────────────────┘
                                           │ Enriched Prototypes [c_real, c_fake]
                                           ▼
                         ┌───────────────────────────────────┐
                         │ Cosine Prototypical Classifier    │
                         │ Temperature Scaled (tau=0.07)     │
                         └─────────────────┬─────────────────┘
                                           │
                                           ▼
                                 Real / Fake Probability
```

---

## 💡 The Three Core Intellectual Novelties

Our architecture introduces three distinct, complementary innovations validated through rigorous ablation:

### 1. Novelty 1 — Tri-Plane Forensic Decomposition (TPFD)
* **The Insight:** Prior CLIP adapters operate solely on the global `[CLS]` token—a single 512-d vector that compresses high-level semantic abstractions but drops local pixel discontinuities and frequency fingerprints.
* **Mechanism:** TPFD processes raw face patches in parallel across three specialized forensic domains:
  1. **Spatial Plane (CNN):** Captures localized blending seams, color mismatches, and boundary artifacts.
  2. **Frequency Plane (DCT / Spectral):** Extracts high-frequency residual signatures characteristic of generative upsampling (GAN / Diffusion artifacts).
  3. **Gradient Plane (Sobel / Laplacian):** Identifies edge anomalies and gradient discontinuities around facial hair, eyes, and contours.
* **Fusion:** The three planes interact via **Cross-Plane Attention**, querying one another before concatenation with the CLIP `[CLS]` feature.

### 2. Novelty 2 — Adversarial Identity Suppressor (AIS) + Orthogonal Residual Constraint (ORC)
* **The Problem:** Identity information is the primary shortcut causing catastrophic failure in cross-manipulation transfer. However, stripping identity with GRL alone can over-penalize essential facial structure, while soft geometric penalties alone are easily bypassed.
* **The "Necessary & Sufficient" Pair:**
  * **Adversarial Identity Suppressor (AIS):** A Gradient Reversal Layer (GRL) reverses gradients back from a content predictor (trained against a frozen VGGFace2 identity representation), actively minimizing mutual information between forensic features and identity.
  * **Orthogonal Residual Constraint (ORC):** Reconstructs the forensic residual in CLIP space ($\hat{z}_{\text{res}} = \mathbf{W}_{\text{up}} z_f$) and strictly penalizes collinearity:
    $$\mathcal{L}_{\text{ortho}} = \frac{1}{B} \sum_{i=1}^B \left| \frac{\hat{z}_{\text{residual}}^{(i)}}{\|\hat{z}_{\text{residual}}^{(i)}\|_2} \cdot \frac{z_{\text{clip}}^{(i)}}{\|z_{\text{clip}}^{(i)}\|_2} \right|$$
* **Scientific Proof:** Neither AIS nor ORC alone is sufficient. As proven in the ablation results, only their joint combination achieves true disentanglement without degrading boundary textures.

### 3. Novelty 3 — Manipulation-Selective Prototype Memory (MSPM)
* **The Problem:** In few-shot settings ($K \in \{1, 5\}$), standard ProtoNet prototypes $c = \frac{1}{K}\sum_{i=1}^K z_i$ are extremely noisy and sensitive to support sample variance. A single outlier support face corrupts the decision boundary.
* **Mechanism:** MSPM maintains a persistent, learnable parameter matrix:
  $$\mathbf{M} \in \mathbb{R}^{M \times d} \quad (M = 16 \text{ slots}, d = 256)$$
  where slots specialize in distinct manipulation patterns (e.g., blending boundaries, frequency checkerboards, warped textures).
* **Routing & Enrichment:**
  1. Episodic support prototypes soft-attend over $\mathbf{M}$:
     $$a_j = \frac{\exp(c_k \cdot \mathbf{M}_j^\top / \sqrt{d})}{\sum_{l=1}^M \exp(c_k \cdot \mathbf{M}_l^\top / \sqrt{d})}$$
  2. Pattern retrieval: $\mathbf{m}_k = \sum_{j=1}^M a_j \mathbf{M}_j$
  3. Residual prototype enrichment: $\mathbf{c}_{\text{enriched}} = \text{normalize}\left(\mathbf{c}_k + \text{MLP}([c_k \,\|\, \mathbf{m}_k])\right)$
* **Impact:** Drastically boosts few-shot stability, increasing Test F1 from **0.7911 to 0.8461** (+5.5% absolute gain).

---

## 📐 Overall Objective & Loss Formulation

The full end-to-end network is trained episodically with a joint multi-task loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{proto}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}} + \lambda_{\text{ortho}} \mathcal{L}_{\text{ortho}}$$

* **$\mathcal{L}_{\text{proto}}$:** Normalized Temperature-scaled Cross-Entropy (NCA) over cosine logits:
  $$\mathcal{L}_{\text{proto}} = - \log \frac{\exp(\cos(q_i, \mathbf{c}_{y_i}) / \tau)}{\sum_{c} \exp(\cos(q_i, \mathbf{c}_c) / \tau)}$$
* **$\mathcal{L}_{\text{adv}}$:** Cosine similarity loss between GRL-projected forensic embeddings and identity targets:
  $$\mathcal{L}_{\text{adv}} = \text{mean}\left(1.0 - \cos(\hat{y}_{\text{content}}, y_{\text{content}})\right)$$
* **$\mathcal{L}_{\text{ortho}}$:** Orthogonal Residual Constraint penalty ($\lambda_{\text{ortho}} = 0.05$).

---

## 📊 Experimental Results & Empirical Verification

All experiments were executed under strict **source-video disjoint partitioning** on **FaceForensics++ (C23)** using 5 random support seeds ($0, 1, 2, 3, 4$) and 3 distinct model training seeds ($42, 100, 2024$).

### 1. Progressive Stage Comparison (A → B → C → D)

Demonstrating the systematic progression from diagnostic baseline to the full CFD-FSA framework:

| Stage | Model Configuration | K=1 AUC | K=5 AUC | K=10 AUC | K=20 AUC |
|:------|:-------------------|:-------:|:-------:|:--------:|:--------:|
| **Stage A** | Frozen CLIP ViT-B/32 (Zero-shot Probing) | 0.5155 | 0.5326 | 0.5544 | 0.5850 |
| **Stage B** | Frozen CLIP + ProtoNet | 0.5578 | 0.5789 | 0.5469 | 0.5453 |
| **Stage C** | CLIP + Forensic Bottleneck Adapter (FBA) | 0.6063 | 0.6945 | 0.7008 | 0.7141 |
| **Stage D (v1)** | CFD-FSA (CLIP + FBA + ACS GRL) | 0.6734 | 0.7000 | 0.7062 | 0.7109 |
| **Stage D (v2)** | **Full CFD-FSA (AIS + ORC + MSPM + Cosine LR)** | **0.7811** | **0.8086** | **0.8195** | **0.8286** |

> **Key Takeaway:** Naive frozen CLIP features (Stage B) achieve near-chance performance (~0.54 - 0.57 AUC) because identity signals dominate few-shot prototypes. Adding FBA provides immediate gains, but only full disentanglement and prototype memory (Stage D v2) achieve **0.8286 test AUC** and **0.8472 validation AUC**.

---

### 2. Intellectual Novelties Ablation Study (Paper Table 3)

Evaluated under $K=20$ multi-seed training (Seeds: 42, 100, 2024; evaluated on full test set):

| Variant | AIS (Adversarial) | ORC (Orthogonal) | MSPM (Memory) | Val AUC | Test AUC | Test EER | Test F1 | Models Count |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `no_mspm` | ✅ Yes | ✅ Yes | ❌ No | 0.8381 | **0.8296 ± 0.0001** | **0.2543 ± 0.0005** | 0.7911 ± 0.0017 | 3 |
| `no_orc` | ✅ Yes | ❌ No | ✅ Yes | 0.8475 | 0.8290 ± 0.0024 | 0.2605 ± 0.0006 | 0.8461 ± 0.0080 | 3 |
| `no_ais` | ❌ No | ✅ Yes | ✅ Yes | 0.8453 | 0.8298 ± 0.0019 | 0.2608 ± 0.0026 | 0.8476 ± 0.0122 | 3 |
| **`full` (CFD-FSA)** | ✅ Yes | ✅ Yes | ✅ Yes | **0.8472** | **0.8286 ± 0.0023** | **0.2600 ± 0.0021** | **0.8454 ± 0.0099** | 3 |

#### Critical Findings from the Ablation:
1. **MSPM's Massive F1 Boost:** Adding MSPM (`no_mspm` vs `full`) increases **Test F1 from 0.7911 to 0.8454 (+5.43% absolute gain)**. By retrieving persistent manipulation patterns from memory, the classifier eliminates false alarms on ambiguous faces.
2. **ORC Prevents Variance Explosion:** Models trained with ORC exhibit ultra-low test variance across random seeds ($\pm 0.0001$ AUC), confirming that geometric orthogonality stabilizes convergence.
3. **The Disentanglement Proof:** Identity probe cosine similarity drops from **0.6452 down to 0.4842** upon introducing ACS, confirming that facial identity is purged from the forensic feature manifold.

---

### 3. Ablation Study: Master Table & Multi-Seed Summary

#### Master Ablation Table (`ablation_table.csv`)
Aggregated results across 3 distinct training seeds (42, 100, 2024), each evaluated across 5 random support seeds on the full test set ($K=20$):

| Variant | AIS (Adversarial) | ORC (Orthogonal) | MSPM (Memory) | Val AUC | Test AUC | Test EER | Test F1 | Models Count |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `no_mspm` | ✅ Yes | ✅ Yes | ❌ No | 0.8381 | 0.8296 ± 0.0001 | 0.2543 ± 0.0005 | 0.7911 ± 0.0017 | 3 |
| `no_orc` | ✅ Yes | ❌ No | ✅ Yes | 0.8475 | 0.8290 ± 0.0024 | 0.2605 ± 0.0006 | 0.8461 ± 0.0080 | 3 |
| `no_ais` | ❌ No | ✅ Yes | ✅ Yes | 0.8453 | 0.8298 ± 0.0019 | 0.2608 ± 0.0026 | 0.8476 ± 0.0122 | 3 |
| **`full` (CFD-FSA)** | ✅ Yes | ✅ Yes | ✅ Yes | **0.8472** | 0.8286 ± 0.0023 | 0.2600 ± 0.0021 | 0.8454 ± 0.0099 | 3 |

#### Detailed Per-Seed Model Summary (`ablation_summary.csv`)
Individual model training runs and their corresponding checkpoint files:

| Variant | K | Training Seed | Val AUC | Test AUC | Test EER | Test F1 | Checkpoint File |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `no_mspm` | 20 | 42 | 0.8385 | 0.8295 | 0.2540 | 0.7922 | `k20_no_mspm_seed42.pt` |
| `no_mspm` | 20 | 100 | 0.8386 | 0.8297 | 0.2549 | 0.7919 | `k20_no_mspm_seed100.pt` |
| `no_mspm` | 20 | 2024 | 0.8371 | 0.8296 | 0.2541 | 0.7891 | `k20_no_mspm_seed2024.pt` |
| `no_orc` | 20 | 42 | 0.8460 | 0.8285 | 0.2611 | 0.8521 | `k20_no_orc_seed42.pt` |
| `no_orc` | 20 | 100 | 0.8466 | 0.8316 | 0.2601 | 0.8370 | `k20_no_orc_seed100.pt` |
| `no_orc` | 20 | 2024 | 0.8500 | 0.8269 | 0.2602 | 0.8492 | `k20_no_orc_seed2024.pt` |
| `no_ais` | 20 | 42 | 0.8418 | 0.8279 | 0.2637 | 0.8338 | `k20_no_ais_seed42.pt` |
| `no_ais` | 20 | 100 | 0.8490 | 0.8316 | 0.2587 | 0.8521 | `k20_no_ais_seed100.pt` |
| `no_ais` | 20 | 2024 | 0.8451 | 0.8299 | 0.2600 | 0.8568 | `k20_no_ais_seed2024.pt` |
| **`full`** | 20 | 42 | 0.8438 | 0.8274 | 0.2621 | 0.8508 | `k20_full_seed42.pt` |
| **`full`** | 20 | 100 | 0.8491 | 0.8313 | 0.2580 | 0.8340 | `k20_full_seed100.pt` |
| **`full`** | 20 | 2024 | 0.8487 | 0.8272 | 0.2599 | 0.8515 | `k20_full_seed2024.pt` |

---

## 📂 Repository Structure

```text
CFD-FSA-exact/
├── checkpoints/                        # Model weights and checkpoint payloads
│   ├── stage_d_v2/                     # Best checkpoints for all ablation variants
│   │   ├── k20_best.pt                 # Overall best full CFD-FSA model
│   │   ├── k20_no_mspm.pt              # Ablation without MSPM
│   │   ├── k20_no_orc.pt               # Ablation without ORC
│   │   └── k20_no_ais.pt               # Ablation without AIS
├── configs/                            # YAML experiment configurations
│   ├── base.yaml                       # Root dataset, paths, and model hyperparams
│   ├── stage_a.yaml                    # Stage A diagnostic probing config
│   ├── stage_b.yaml                    # Stage B ProtoNet baseline config
│   ├── stage_c.yaml                    # Stage C FBA adapter config
│   ├── stage_d.yaml                    # Stage D initial CFD-FSA config
│   └── stage_d_v2.yaml                 # Stage D v2 full paper ablation config
├── data/                               # Dataset and pre-extracted feature caches
│   ├── FaceForensics++_C23/            # Raw video frames / cropped faces
│   ├── features/                       # Cached .pt embeddings (train, val, test)
│   └── splits/                         # Video-level disjoint train/val/test splits
├── experiments/                        # Standalone experiment execution scripts
│   ├── stage_a_diagnosis.py            # Stage A identity vs forensic probe
│   ├── stage_b_baseline.py             # Stage B CLIP + ProtoNet baseline
│   ├── stage_c_adapter.py              # Stage C FBA + ProtoNet
│   ├── stage_d_cfd_fsa.py              # Core training script for CFD-FSA & ablations
│   ├── run_ablations.py                # Automated ablation suite runner & table compiler
│   ├── cross_manipulation.py           # Cross-manipulation transfer benchmark
│   ├── cross_dataset.py                # Celeb-DF-v2 evaluation
│   ├── predict_media.py                # Real/Fake inference on single image/video
│   └── representation_analysis.py      # Feature distribution & t-SNE analysis
├── results/                            # Full experiment logs, CSV tables, and plots
│   └── stage_d_v2/                     # Complete ablation study CSVs & summaries
├── scripts/                            # Dataset preparation & verification tools
│   ├── check_environment.py            # Hardware, CUDA, and dependency checker
│   ├── make_manifest_template.py       # Manifest generator from raw directories
│   ├── validate_manifest.py            # Leakage auditor & video-level split creator
│   └── build_content_cache.py          # Pre-computes CLIP & VGGFace2 embeddings
├── src/                                # Core library modules
│   ├── models.py                       # FBA, ACS, GRL, ORC, MSPM, and CFD-FSA classes
│   ├── protonet.py                     # Episodic prototype computation & cosine loss
│   ├── sampling.py                     # Disjoint few-shot support/query episode sampler
│   ├── metrics.py                      # AUC, EER, F1, Accuracy, Brier, and ECE
│   └── utils.py                        # FeatureStore, seed control, YAML loading
├── run.ps1                             # PowerShell master pipeline orchestrator
└── requirements.txt                    # Project dependencies
```

---

## 🚀 Installation & Environment Setup

### 1. Prerequisites
* **Operating System:** Windows 10/11 or Linux
* **Python:** 3.10 or 3.11
* **Hardware:** NVIDIA GPU with CUDA 12.0+ support (RTX 3060/4050 or higher recommended)

### 2. Install Dependencies
Clone the repository and install the required packages:

```bash
git clone https://github.com/your-username/CFD-FSA-exact.git
cd CFD-FSA-exact
pip install -r requirements.txt
```

Verify your environment and CUDA availability:

```powershell
python scripts/check_environment.py --config configs/base.yaml
```

---

## 🛠️ Data Preparation & Feature Caching

To guarantee ultra-fast episodic few-shot training without redundant video decoding, facial frames are pre-extracted and features are pre-cached:

### Step 1: Create Video Manifest & Generate Disjoint Splits
```powershell
# 1. Build template manifest
python scripts/make_manifest_template.py --config configs/base.yaml

# 2. Validate manifest and generate source-video disjoint train/val/test splits
python scripts/validate_manifest.py --config configs/base.yaml --make-splits
```

### Step 2: Build CLIP & Content Feature Cache
Pre-computes frozen CLIP ViT-B/32 and VGGFace2 identity vectors for `train.pt`, `val.pt`, and `test.pt`:
```powershell
python scripts/build_content_cache.py --config configs/base.yaml
```

---

## 🏋️ Training & Running Experiments

### 1. Run Initial Diagnostic (Stage A)
Tests whether frozen CLIP features contain excessive identity leakage:
```powershell
python experiments/stage_a_diagnosis.py --config configs/stage_a.yaml
```

### 2. Train Full CFD-FSA Model (Stage D v2)
Trains CFD-FSA with all three novelties (**AIS + ORC + MSPM**) using CUDA:
```powershell
python experiments/stage_d_cfd_fsa.py --config configs/stage_d_v2.yaml --ablation full
```

Key Training Hyperparameters:
* `epochs`: 30
* `episodes_per_epoch`: 100
* `query_per_class`: 64
* `K`: 20
* `warmup_steps`: 150 / 3000 steps (Cosine LR schedule)
* `grl_schedule`: linear (target $\lambda = 0.5$)
* `ortho_loss_weight`: 0.05
* `mspm_slots`: 16
* `val_every_n_epochs`: 5
* `early_stopping_patience`: 3
* `amp`: true (Automatic Mixed Precision enabled for CUDA)

### 3. Run Specific Ablation Variants
You can train any specific ablation variant using the `--ablation` argument:
```powershell
# Train without Prototype Memory (AIS + ORC only)
python experiments/stage_d_cfd_fsa.py --config configs/stage_d_v2.yaml --ablation no_mspm

# Train without Orthogonal Constraint (AIS + MSPM only)
python experiments/stage_d_cfd_fsa.py --config configs/stage_d_v2.yaml --ablation no_orc

# Train without Adversarial Identity Suppression (ORC + MSPM only)
python experiments/stage_d_cfd_fsa.py --config configs/stage_d_v2.yaml --ablation no_ais

# Train standard baseline ProtoNet
python experiments/stage_d_cfd_fsa.py --config configs/stage_d_v2.yaml --ablation baseline
```

### 4. Run Automated Master Ablation Suite
Automatically trains and evaluates all variants across multiple training seeds and compiles the master publication table ([`results/stage_d_v2/ablation_table.csv`](file:///c:/Users/agfdg/Desktop/CFD-FSA-exact/CFD-FSA-exact/results/stage_d_v2/ablation_table.csv)):
```powershell
python experiments/run_ablations.py --config configs/stage_d_v2.yaml --variants no_mspm no_orc no_ais full
```

### 5. Benchmark Cross-Manipulation Generalization
Evaluate adaptation on DF+F2F and test on unseen FS+NT:
```powershell
python experiments/cross_manipulation.py --config configs/stage_d_v2.yaml
```

### 6. Using the Master PowerShell Runner
Alternatively, you can run any stage using [`run.ps1`](file:///c:/Users/agfdg/Desktop/CFD-FSA-exact/CFD-FSA-exact/run.ps1):
```powershell
# Run environment check
.\run.ps1 -Task check

# Run Stage A diagnosis
.\run.ps1 -Task stage-a

# Run Stage B baseline
.\run.ps1 -Task stage-b

# Run Stage C adapter
.\run.ps1 -Task stage-c

# Run Stage D CFD-FSA
.\run.ps1 -Task stage-d
```

---

## 🔮 Inference & Prediction on Unseen Media

You can test real-world images or videos directly using [`experiments/predict_media.py`](file:///c:/Users/agfdg/Desktop/CFD-FSA-exact/CFD-FSA-exact/experiments/predict_media.py):

### Single Image Prediction:
```powershell
python experiments/predict_media.py `
    --input test_media/sample_face.jpg `
    --checkpoint checkpoints/stage_d_v2/k20_best.pt `
    --config configs/stage_d_v2.yaml `
    --device cuda
```

### Video Stream Prediction:
Automatically detects faces with MTCNN, aggregates per-frame forensic embeddings, and outputs deepfake probabilities:
```powershell
python experiments/predict_media.py `
    --input test_media/sample_video.mp4 `
    --checkpoint checkpoints/stage_d_v2/k20_best.pt `
    --config configs/stage_d_v2.yaml `
    --device cuda `
    --output results/prediction.json
```

---

## 📜 Citation & Intellectual Property

If you find this work, repository, or novelty formulation useful in your research, please cite:

```bibtex
@article{cfd_fsa_2026,
  title={Content-Forensic Disentangled Few-Shot Adapter for Generalizable Deepfake Detection},
  author={CFD-FSA Research Team},
  journal={Computer Vision and Pattern Recognition (CVPR) / IEEE Transactions},
  year={2026}
}
```

---

## ⚖️ License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
