# EA-ViT
Official inplementation of EA-ViT: Efficient Adaptation for Elastic Vision Transformer [ICCV 2025]

## 📝 Abstract
Vision Transformers (ViTs) have emerged as a foundational model in computer vision, excelling in generalization and adaptation to downstream tasks. However, deploying ViTs to support diverse resource constraints typically requires retraining multiple, size-specific ViTs, which is both time-consuming and energy-intensive.
To address this issue, we propose an efficient ViT adaptation framework that enables a single adaptation process to generate multiple models of varying sizes for deployment on platforms with various resource constraints.
Our approach comprises two stages. In the first stage, we enhance a pre-trained ViT with a nested elastic architecture that enables structural flexibility across embedding dimension, number of attention heads, MLP expansion ratio, and network depth. To preserve pre-trained knowledge and ensure stable adaptation, we adopt a curriculum-based training strategy that progressively increases elasticity. In the second stage, we design a lightweight router to select submodels according to computational budgets and downstream task demands. Initialized with Pareto-optimal configurations derived via a customized NSGA-II algorithm, the router is then jointly optimized with the backbone.
Extensive experiments on multiple benchmarks demonstrate the effectiveness and versatility of EA-ViT. 


## 📦 Environment Setup

```bash
# 1) Create and activate a fresh conda environment (Python 3.9)
conda create -n EA-ViT python=3.9
conda activate EA-ViT

# 2) Clone the EA-ViT repository
git clone https://github.com/zcxcf/EA-ViT
cd EA-ViT

# 3) Install all Python dependencies listed in requirements.txt
pip install -r requirements.txt
```
## ⚡ Quick Start

### 0. Download Rearranged Checkpoint
Grab the checkpoint **re-ordered by importance** from [Google Drive](https://drive.google.com/file/d/1f1ku-vQlzGDwGPr9FAMsbrVGgg03ocBT/view?usp=drive_link) and place it in `./pretrained_para/`.

### 1. Datasets

All supported datasets are defined in `./dataloader/image_datasets.py`.

### 2. Training & Search Pipeline

```bash
# Stage 1 training
bash stage1.sh
# → checkpoints saved to ./logs_weight

# Search a promising sub-model to initialise the router
bash search_submodel.sh
# → sub-models stored in ./NSGA

# Stage 2 training
bash stage2.sh
# → checkpoints saved to ./logs_weight

```
### 3. Evaluation
```bash
# Edit inference.sh so that CHECKPOINT points to the weights you want to evaluate
bash inference.sh
```

## ✏️ Citation
If you find EA-ViT is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex

```



