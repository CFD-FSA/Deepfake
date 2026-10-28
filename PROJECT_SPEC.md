CFD-FSA — FINAL PROJECT DETAILS & TEAM WORK DIVISION
========================================================

PROJECT TITLE
-------------
Content-Forensic Disentanglement for Few-Shot Deepfake Detection

PROJECT TYPE
------------
Deep Learning / Computer Vision / Deepfake Detection / Few-Shot Learning

RESEARCH GOAL
-------------
We want to investigate whether foundation-model features used for few-shot
deepfake detection contain content-related information (such as identity or
facial appearance) that can become a shortcut instead of manipulation-specific
forensic information.

Main research question:
"Can explicitly suppressing content-related information from a
foundation-model representation improve few-shot deepfake detection and its
generalization to unseen manipulation types?"

IMPORTANT:
This is a hypothesis to be experimentally validated. We must not assume that
content suppression will improve results before running the experiments.


1. CORE PROPOSED METHOD
-----------------------

Method name:
CFD-FSA = Content-Forensic Disentangled Few-Shot Adapter

Pipeline:

Face Image
    |
    v
Frozen CLIP ViT-B/32
    |
    v
Forensic Bottleneck Adapter (FBA)
    |
    v
Adversarial Content Suppression (ACS)
    |
    v
Forensic Representation
    |
    v
Few-Shot Forensic Prototypical Classifier (ProtoNet)
    |
    v
Real / Fake prediction


Main components:

A. Frozen Foundation Encoder (FFE)
   - CLIP ViT-B/32
   - Completely frozen during adaptation.
   - Provides general visual representation.

B. Forensic Bottleneck Adapter (FBA)
   - Lightweight trainable bottleneck.
   - Maps CLIP feature dimension d to compact forensic dimension r.
   - Purpose: adapt foundation features without fine-tuning the entire CLIP.

C. Content Representation Predictor (CRP)
   - Uses a frozen face-recognition/identity representation as an operational
     proxy for content information.
   - Identity is only a proxy for one component of content; it is NOT claimed
     to represent all possible content information.

D. Adversarial Content Suppressor (ACS)
   - Uses a Gradient Reversal Layer (GRL).
   - Content predictor tries to predict the content target.
   - GRL makes the forensic representation less predictive of that content.
   - The goal is to suppress content-correlated information while retaining
     manipulation-specific information.

E. Few-Shot Forensic Prototypical Classifier (FFPC)
   - Uses prototypes constructed from few-shot support examples.
   - Classification is performed using distance/cosine similarity in the
     learned forensic space.


2. RESEARCH CONTRIBUTION
------------------------

We are NOT claiming that CLIP, LoRA/adapters, GRL, ProtoNet, or adversarial
learning are individually novel.

The intended contribution is:

1. A systematic investigation of content-forensic entanglement in
   foundation-model-based few-shot deepfake detection.

2. A lightweight adversarial forensic adaptation framework that learns a
   compact representation while suppressing content-related information.

3. Controlled evaluation of whether content suppression improves
   generalization to manipulation types excluded during adaptation.

4. Analysis of the trade-off between content predictability and forensic
   discriminability.

The contribution must be supported by experiments. If the Stage-A diagnosis
does not support the hypothesis, the formulation must be reconsidered rather
than forcing a positive conclusion.


3. DATASETS
-----------

PRIMARY DATASET:
FaceForensics++

Core manipulation types:
- Deepfakes (DF)
- Face2Face (F2F)
- FaceSwap (FS)
- NeuralTextures (NT)
- Original = real videos

INITIAL VERSION:
- C23 / H.264 compressed videos
- Start with C23 because it is manageable and sufficient to build the main
  pipeline.

LATER:
- C0
- C40

These will be used for compression robustness experiments.

SECONDARY DATASET:
Celeb-DF-v2

Purpose:
- Cross-dataset generalization.
- Train/adapt on FF++ and evaluate on Celeb-DF-v2.

NOT REQUIRED FOR THE INITIAL IMPLEMENTATION:
- DFDC
- WildDeepfake
- DeeperForensics-1.0
- Separate DeepFakeDetection dataset
- FaceShifter

These may be considered only if time/resources permit after the core study.


4. DATA PROCESSING PIPELINE
---------------------------

Raw videos
    |
    v
Frame extraction
    |
    v
Face detection / face cropping
    |
    v
Metadata generation
    |
    v
Leakage audit
    |
    v
Train / validation / test split
    |
    v
Feature extraction / caching
    |
    v
Experiments


IMPORTANT DATA RULES
--------------------
- Split by source video, NOT randomly by frame.
- Frames from the same source video must never appear in both adaptation and
  evaluation.
- Few-shot support examples must come only from the training/adaptation set.
- Test data must remain untouched during hyperparameter selection.
- Manipulation labels must be preserved.
- Compression labels must be preserved.
- Do not invent video IDs or identity labels from filenames unless supported
  by dataset metadata.
- Report identity overlap where the metadata permits it.


5. EXPERIMENTAL STAGES
----------------------

STAGE A — CONTENT-FORENSIC ENTANGLEMENT DIAGNOSIS
-------------------------------------------------
Goal:
Determine whether frozen foundation-model features contain measurable
content-related information and how this relates to forensic classification.

Tasks:
- Extract frozen CLIP features.
- Train independent content probes.
- Train independent forensic probes.
- Compare performance across few-shot settings.
- Evaluate content predictability and forensic discriminability.

Few-shot settings:
K = 1, 5, 10, 20, 100, and all where meaningful.

Important:
The content probe must be designed carefully. A simple closed-set identity
classifier can be misleading if evaluation identities are unseen. Consider
identity retrieval / verification or another open-set-safe probe.


STAGE B — FROZEN CLIP + PROTONET BASELINE
------------------------------------------
Pipeline:
CLIP -> ProtoNet

Purpose:
Establish the basic few-shot foundation-model baseline.


STAGE C — CLIP + FBA + PROTONET
-------------------------------
Pipeline:
CLIP -> FBA -> ProtoNet

Purpose:
Measure how much improvement comes from the bottleneck adapter alone.


STAGE D — PROPOSED CFD-FSA
--------------------------
Pipeline:
CLIP -> FBA -> ACS -> ProtoNet

Purpose:
Test whether adversarial content suppression provides additional benefit
over the adapter baseline.


6. MAIN GENERALIZATION EXPERIMENT
---------------------------------

Cross-manipulation generalization is a major experiment.

Example:

ADAPT/TRAIN:
- Deepfakes + Face2Face

TEST:
- FaceSwap + NeuralTextures

Then reverse:

ADAPT/TRAIN:
- FaceSwap + NeuralTextures

TEST:
- Deepfakes + Face2Face

More train/test manipulation combinations can be evaluated if computational
budget permits.

Important:
The test manipulation types must be excluded from adaptation when measuring
unseen-manipulation generalization.


7. FEW-SHOT PROTOCOL
--------------------

K-shot settings:
- 1-shot
- 5-shot
- 10-shot
- 20-shot

For each relevant training category:
- K real support examples
- K fake support examples

Use multiple support splits/seeds because few-shot results can be unstable.

No source-video overlap between support/adaptation and evaluation.


8. CROSS-DATASET EXPERIMENT
---------------------------

Train/adapt:
FaceForensics++

Evaluate:
Celeb-DF-v2

Purpose:
Test whether the learned representation transfers beyond FF++.


9. COMPRESSION ROBUSTNESS
-------------------------

Main initial training:
C23

Later evaluation:
- C0
- C23
- C40

Example:
Train/adapt on C23
Test on C0, C23, and C40

Do NOT claim that the method solves compression problems unless the experiments
actually support that conclusion.


10. METRICS
-----------

Primary:
- ROC-AUC
- Accuracy
- EER
- TPR/TAR at low FAR
- Precision
- Recall
- F1

For representation/disentanglement:
- Content probe performance
- Independent forensic probe performance
- Representation similarity/visualization

Secondary:
- ECE / calibration
- Reliability diagrams

For few-shot experiments:
- Mean and standard deviation across multiple seeds/support splits.
- Confidence intervals where practical.

Do not report only one lucky run.


11. REQUIRED ABLATIONS
----------------------

A. CLIP + ProtoNet
B. CLIP + FBA + ProtoNet
C. CLIP + FBA + ACS + ProtoNet (PROPOSED)
D. CLIP + ACS without bottleneck
E. FBA + standard classifier
F. FBA + ProtoNet without ACS
G. Different ACS/GRL lambda values
H. Different bottleneck dimensions r
I. Different content targets
J. Content probe before vs after ACS
K. Independent forensic probe
L. Frequency branch ON/OFF as a secondary ablation

The purpose is to show where any improvement comes from.


12. BASELINES
------------

At minimum:
- Frozen CLIP + standard classifier / ProtoNet
- CLIP + FBA
- Proposed CLIP + FBA + ACS

For stronger comparison, implement/report established deepfake detection
baselines where reproducible and computationally feasible.

Potential literature baselines:
- RECURVE
- ASFormer
- FACT
- DiffAct

These should only be included if their implementations and input protocols
are appropriate to the exact task.


13. REQUIRED ANALYSIS
---------------------

FIGURE 1:
Motivation — entangled representation vs content-suppressed representation.

FIGURE 2:
Stage-A content/forensic probe analysis.

FIGURE 3:
Complete CFD-FSA architecture.

FIGURE 4:
Few-shot generalization results.

FIGURE 5:
Cross-manipulation performance heatmap.

FIGURE 6:
Representation analysis (t-SNE/UMAP or equivalent).

FIGURE 7:
Failure-case analysis.

Failure analysis:
- At least 10 meaningful examples.
- Compare baseline vs CFD-FSA.
- Inspect whether the model is using content-related shortcuts.
- Include successful and failed cases rather than only favorable examples.


14. TEAM WORK DIVISION
----------------------

PERSON 1 — DATASET & DATA ENGINEERING
--------------------------------------
Responsibilities:
- Download/organize FF++ C23.
- Verify all required folders.
- Understand CSV/metadata.
- Extract frames.
- Face detection/cropping.
- Create final manifest.
- Preserve:
  path
  label
  video_id
  manipulation
  compression
  identity (only when supported)
- Build source-video split.
- Run leakage audit.
- Prepare train/validation/test metadata.

Deliverables:
- Clean dataset structure
- manifest.csv
- split files
- leakage_report.txt/json
- dataset statistics


PERSON 2 — FOUNDATION MODEL & FEATURE PIPELINE
-----------------------------------------------
Responsibilities:
- Implement frozen CLIP ViT-B/32.
- Verify ALL CLIP parameters are frozen.
- Implement preprocessing.
- Extract and cache CLIP features.
- Benchmark feature extraction speed/memory.
- Ensure reproducibility.

Deliverables:
- feature extraction script
- cached feature format
- environment/setup documentation
- sanity-check report


PERSON 3 — BASELINES & FEW-SHOT PROTOCOL
-----------------------------------------
Responsibilities:
- Implement CLIP + ProtoNet.
- Implement CLIP + FBA + ProtoNet.
- Implement standard classifier comparison.
- Implement K-shot support sampling.
- Implement multiple support seeds.
- Produce baseline metrics.

Deliverables:
- baseline training/evaluation scripts
- support/query episode generation
- baseline result tables
- reproducible experiment commands


PERSON 4 — CONTENT PROBE & ACS / CFD-FSA
-----------------------------------------
Responsibilities:
- Implement content representation pipeline.
- Implement Content Representation Predictor.
- Implement Gradient Reversal Layer.
- Implement Adversarial Content Suppression.
- Implement CFD-FSA training objective.
- Test lambda sensitivity.
- Compare content prediction before/after ACS.
- Verify that forensic performance is retained.

Deliverables:
- ACS module
- CFD-FSA model
- loss implementation
- content-probe experiments
- ablation results


PERSON 5 — EVALUATION, GENERALIZATION & ANALYSIS
------------------------------------------------
Responsibilities:
- Cross-manipulation experiments.
- FF++ -> Celeb-DF-v2 evaluation.
- Compression experiments C0/C23/C40.
- Statistical analysis.
- ROC/AUC/EER/TAR@FAR.
- t-SNE/UMAP/heatmaps.
- Failure analysis.
- Prepare final tables and figures.

Deliverables:
- evaluation scripts
- final metrics
- figures
- statistical reports
- failure-case analysis


IF TEAM HAS FEWER THAN 5 PEOPLE
--------------------------------
Combine roles as follows:

3 people:
Person 1 = Dataset + preprocessing
Person 2 = Model + baselines + CFD-FSA
Person 3 = Evaluation + analysis + paper experiments

4 people:
Person 1 = Dataset
Person 2 = CLIP + baselines
Person 3 = CFD-FSA + content probe
Person 4 = Evaluation + analysis

If one person is responsible for multiple modules, use Git branches and
clearly defined interfaces.


15. REPOSITORY STRUCTURE
------------------------

CFD-FSA/
|
├── cfd_fsa/
│   ├── data.py
│   ├── models.py
│   ├── protonet.py
│   ├── sampling.py
│   ├── metrics.py
│   └── utils.py
|
├── experiments/
│   ├── stage_a_diagnosis.py
│   ├── stage_b_baseline.py
│   ├── stage_c_adapter.py
│   ├── stage_d_cfd_fsa.py
│   └── cross_manipulation.py
|
├── scripts/
│   ├── check_environment.py
│   ├── validate_manifest.py
│   ├── make_manifest_template.py
│   └── build_content_cache.py
|
├── data/
├── results/
├── configs/
├── requirements.txt
├── README.md
└── run.ps1


16. CODING RULES
----------------

- Do not directly modify main without testing.
- Use separate Git branches for major features.
- Every experiment must have a config/command that can reproduce it.
- Fix random seeds where applicable.
- Save checkpoints and configuration with every experiment.
- Save metrics in machine-readable CSV/JSON.
- Never manually type final results into tables.
- Do not use test data for hyperparameter tuning.
- Document every dataset preprocessing step.
- Do not silently change preprocessing between models.
- Every major result should be reproducible by another team member.


17. TRAINING STRATEGY
---------------------

Do NOT start the complete expensive experiment immediately.

Order:

1. Environment verification
2. Dataset verification
3. Leakage audit
4. Small-data sanity run
5. Stage-A diagnosis
6. CLIP + ProtoNet baseline
7. FBA baseline
8. CFD-FSA implementation
9. CFD-FSA sanity run
10. Full few-shot experiments
11. Cross-manipulation experiments
12. Compression experiments
13. Celeb-DF cross-dataset evaluation
14. Ablations
15. Statistical analysis
16. Final figures/tables
17. Paper writing


18. IMPORTANT CHECKS BEFORE FULL GPU TRAINING
---------------------------------------------

Before spending GPU time, verify:

[ ] Correct FF++ dataset
[ ] Correct four manipulation types
[ ] Correct real/original videos
[ ] C23 compression confirmed
[ ] Metadata available
[ ] Source-video IDs preserved
[ ] No train/test video leakage
[ ] No frame-level leakage
[ ] CLIP completely frozen
[ ] Correct image preprocessing
[ ] Content target verified
[ ] Content probe evaluation protocol is valid
[ ] Few-shot sampling is reproducible
[ ] Test set is untouched during tuning
[ ] Checkpoint saving works
[ ] Metrics logging works


19. EXPECTED FINAL PAPER CLAIM
------------------------------

The paper should ultimately answer:

"Does explicitly suppressing content-related information from foundation-model
representations improve few-shot deepfake detection and generalization to
unseen manipulation types?"

Possible outcomes:

A. If CFD-FSA improves generalization while reducing content predictability:
   this supports the hypothesis.

B. If content predictability decreases but forensic performance also drops:
   the method may be removing useful information; analyze the trade-off.

C. If there is no meaningful change:
   report that honestly and investigate why.

D. If Stage-A diagnosis does not show the expected entanglement:
   reconsider the research hypothesis before claiming disentanglement as a
   solution.

We must let the experiments determine the conclusion.


20. IMMEDIATE NEXT TASKS
------------------------

TASK 1:
Verify the Kaggle FF++ C23 dataset structure and CSV metadata.

TASK 2:
Download only:
- original
- Deepfakes
- Face2Face
- FaceSwap
- NeuralTextures

Do not initially download:
- FaceShifter
- DeepFakeDetection

TASK 3:
Build the dataset manifest and leakage audit.

TASK 4:
Run Stage-A diagnosis.

TASK 5:
Only after Stage A is validated, begin full CFD-FSA implementation/training.


21. SUCCESS CRITERIA
--------------------

The project is considered technically ready for final experiments only when:

- Dataset is leakage-controlled.
- Stage-A diagnosis is valid.
- Baselines are reproducible.
- CFD-FSA is implemented correctly.
- FBA-only and ACS ablations exist.
- Few-shot support sampling is controlled.
- Cross-manipulation testing is strict.
- Cross-dataset testing is performed if resources permit.
- Multiple seeds/support splits are reported.
- Results include uncertainty/statistical analysis.
- Failure cases are analyzed.
- No claim is based on a single run.


FINAL TEAM MESSAGE
------------------

This project is not simply "CLIP + adapter + GRL".

The central research question is whether content-forensic entanglement is a
problem in few-shot foundation-model deepfake detection, and whether explicitly
suppressing content-related information can produce a more manipulation-specific
forensic representation.

Our implementation and experiments must be designed to prove or disprove this
hypothesis, with strict leakage control and strong ablations.

First priority:
DATASET -> LEAKAGE AUDIT -> STAGE-A DIAGNOSIS -> BASELINES -> CFD-FSA ->
GENERALIZATION -> ABLATIONS -> FINAL ANALYSIS.
