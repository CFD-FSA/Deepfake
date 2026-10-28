# CFD-FSA: Implementation Plan for Novelties 2 & 3
## Orthogonal Residual Constraint (ORC) & Manipulation-Selective Prototype Memory (MSPM)

---

## 1. Overview & Motivation

Following the intellectual anchor document ([`CFD-FSA-Novelty-Statement.md`](file:///c:/Users/agfdg/Desktop/CFD-FSA-exact/CFD-FSA-exact/CFD-FSA-Novelty-Statement.md)), the current CFD-FSA model has already achieved **0.8389 validation AUC** and **0.8293 test AUC** with the High & Medium Impact optimizations.

To elevate this to top-tier CVPR/IEEE publishable standard and complete the required ablation table, we implement:
1. **Novelty 2: Orthogonal Residual Constraint (ORC)** paired with **Adversarial Identity Suppression (AIS)**.
2. **Novelty 3: Manipulation-Selective Prototype Memory (MSPM)** with learnable pattern routing.

---

## 2. Mathematical Formulation & Architecture

### Novelty 2: Orthogonal Residual Constraint (ORC)

#### The Problem
Using Adversarial Content Suppression (ACS/GRL) alone runs the risk of either over-suppressing necessary boundary textures or leaving residual identity correlations in unmonitored dimensions. Soft penalties alone without adversarial gradients are easily bypassed.

#### The Mathematical Formulation
Let $z_{\text{clip}} \in \mathbb{R}^D$ be the frozen CLIP feature vector, and let $z_f = \text{FBA}(z_{\text{clip}}) \in \mathbb{R}^d$ be the forensic embedding. The reconstructed residual vector in CLIP space is $\hat{z}_{\text{residual}} = \mathbf{W}_{\text{up}} z_f \in \mathbb{R}^D$.

The **Orthogonal Residual Constraint (ORC)** penalizes any collinearity between the forensic manifold and the semantic CLIP representation:

$$\mathcal{L}_{\text{ortho}} = \frac{1}{B} \sum_{i=1}^B \left| \frac{\hat{z}_{\text{residual}}^{(i)}}{\|\hat{z}_{\text{residual}}^{(i)}\|_2} \cdot \frac{z_{\text{clip}}^{(i)}}{\|z_{\text{clip}}^{(i)}\|_2} \right|$$

#### Combined Loss
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{proto}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}} + \lambda_{\text{ortho}} \mathcal{L}_{\text{ortho}}$$

* Default $\lambda_{\text{ortho}} = 0.05$ (configurable in `cfg["training"]["ortho_loss_weight"]`).
* When $\lambda_{\text{ortho}} = 0$, ORC is disabled (for the ablation table).

---

### Novelty 3: Manipulation-Selective Prototype Memory (MSPM)

#### The Problem
In few-shot learning (especially $K \le 5$), episodic prototypes $c = \frac{1}{K}\sum_{i=1}^K z_i$ are extremely sensitive to the particular faces sampled. A single atypical support face skews the prototype, causing high variance.

#### The Architecture
MSPM introduces a persistent, learnable parameter matrix:
$$\mathbf{M} \in \mathbb{R}^{M \times d} \quad (M = 16 \text{ slots}, d = 256)$$

Each slot in $\mathbf{M}$ learns a persistent signature across manipulation artifacts (e.g. boundary artifacts, GAN frequency patterns, compression artifacts).

```
                 Episodic Prototype c_k (256-d)
                             │
                             ├──────────────────────────────────────┐
                             │                                      │
                             ▼                                      │
             [Soft-Attention over M=16 Slots]                       │
              a_j = softmax(c_k · M_j^T / sqrt(d))                  │
                             │                                      │
                             ▼                                      │
                  Retrieved Pattern: m_k = sum(a_j * M_j)           │
                             │                                      │
                             ▼                                      │
                  Concat [c_k || m_k] (512-d)                       │
                             │                                      │
                             ▼                                      │
                  Linear + LayerNorm + GELU                         │
                             │                                      │
                             ▼                                      │
                         Residual: + c_k ◄──────────────────────────┘
                             │
                             ▼
                  L2 Normalize -> c_enriched (256-d)
```

1. **Routing:** Soft-attention weight $a_j = \frac{\exp(c_k \cdot \mathbf{M}_j^\top / \sqrt{d})}{\sum_{l=1}^M \exp(c_k \cdot \mathbf{M}_l^\top / \sqrt{d})}$
2. **Retrieval:** $\mathbf{m}_k = \sum_{j=1}^M a_j \mathbf{M}_j$
3. **Enrichment:** $\mathbf{c}_{\text{enriched}} = \text{normalize}\left(\mathbf{c}_k + \text{MLP}([\mathbf{c}_k \,\|\, \mathbf{m}_k])\right)$

Query embeddings $q_z$ are classified against the **enriched prototypes** $[\mathbf{c}_{0, \text{enriched}}, \mathbf{c}_{1, \text{enriched}}]$ via temperature-scaled cosine distance.

---

## 3. Implementation Steps

### Step 1: Update `src/models.py`
1. Expose `residual_embedding(clip_features)` on `ForensicBottleneckAdapter` and `CFD_FSA` to enable computing $\mathcal{L}_{\text{ortho}}$.
2. Implement `ManipulationSelectivePrototypeMemory` (M=16, d=forensic_dim).
3. Integrate MSPM into `CFD_FSA` with a toggle `use_mspm: bool = False`.

### Step 2: Update `src/protonet.py`
1. Allow `prototypical_loss` and `prototypical_logits` to optionally accept an `mspm` module or perform memory enrichment on prototypes before computing cosine logits.

### Step 3: Update `experiments/stage_d_cfd_fsa.py`
1. Compute $\mathcal{L}_{\text{ortho}}$ on support and query representations during training when `ortho_loss_weight > 0`.
2. Add CLI flags and config support:
   - `--use-mspm` / `cfg["model"]["use_mspm"]`
   - `--ortho-weight` / `cfg["training"]["ortho_loss_weight"]`
3. Support the required paper ablation table:
   - Baseline: ProtoNet
   - + AIS only (current Stage D v2)
   - + ORC only
   - + AIS + ORC
   - + AIS + ORC + MSPM (Full CFD-FSA)

### Step 4: Verification & Benchmarking
Run end-to-end multi-seed evaluation on FF++ C23 to measure the performance gain from AIS + ORC + MSPM.
