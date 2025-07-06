# EA-ViT
EA-ViT: Efficient Adaptation for Elastic Vision Transformer [ICCV 2025]

## Abstract
Vision Transformers (ViTs) have emerged as a foundational model in computer vision, excelling in generalization and adaptation to downstream tasks. However, deploying ViTs to support diverse resource constraints typically requires retraining multiple, size-specific ViTs, which is both time-consuming and energy-intensive.
To address this issue, we propose an efficient ViT adaptation framework that enables a single adaptation process to generate multiple models of varying sizes for deployment on platforms with various resource constraints.
Our approach comprises two stages. In the first stage, we enhance a pre-trained ViT with a nested elastic architecture that enables structural flexibility across embedding dimension, number of attention heads, MLP expansion ratio, and network depth. To preserve pre-trained knowledge and ensure stable adaptation, we adopt a curriculum-based training strategy that progressively increases elasticity. In the second stage, we design a lightweight router to select submodels according to computational budgets and downstream task demands. Initialized with Pareto-optimal configurations derived via a customized NSGA-II algorithm, the router is then jointly optimized with the backbone.
Extensive experiments on multiple benchmarks demonstrate the effectiveness and versatility of EA-ViT. 


## Environment Setup

```bash
# create and activate env
conda create -n EAViT python=3.9
conda activate EAViT

# install project dependencies
pip install -r requirements.txt
```

## Quick Start
you can download the checkpoint rearranged by importance at [Google Drive]
```bash
# stage1 training
bash stage1.sh

# searching promising submodel to initialize router
bash search_submodel.sh

# stage2 training
bash stage2.sh

```

## Evaluation
```bash

bash inference.sh
```


## Citation
If you find EA-ViT is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex

```
