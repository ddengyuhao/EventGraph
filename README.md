# EventGraph-LMM: Submodular Subgraph Selection for Token-Efficient Long-Video Understanding

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10-green.svg)]()
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)]()

</div>

## 📖 Abstract

Understanding long-form videos with **Large Multimodal Models (LMMs)** is challenging due to massive token counts and limited context windows. We propose **EventGraph-LMM**, a training-free framework that reformulates long-video compression as a constrained **subgraph selection problem**. 

By modeling videos as directed event graphs and applying monotone submodular maximization (via the **CELF algorithm**), we efficiently select the most informative frames. Furthermore, we introduce a **Graph-Constrained Chain-of-Thought (Topo-CoT)** mechanism that grounds the LLM's reasoning path in the constructed graph structure.

### 🚀 Key Features

- **📉 Token-Efficiency**: Reduces visual tokens by **60-80%** while maintaining or exceeding full-context performance.
- **🧩 Event Graph Structure**: Models video as a graph $G=(V, E)$ where nodes are events and edges represent semantic/temporal dependencies.
- **⚡ Fast Selection**: Utilizes the **CELF** algorithm for submodular maximization, ensuring theoretical approximation guarantees with low latency.
- **🧠 Graph-Injected Reasoning**: Introduces **Topo-CoT** to guide the LLM's reasoning along verified semantic paths.
- **🔌 Multi-Backbone Support**: Seamlessly supports **Video-LLaVA**, **Qwen2.5-VL**, and **LLaVA-NeXT**.

---

## 🛠️ Framework

The **EventGraph Pipeline** consists of three stages:

1.  **Event Graph Construction**: We decompose the video into events (shots) using TransNetV2 and compute semantic similarities (via CLIP) to build a directed graph.
2.  **Subgraph Selection**: We solve the submodular maximization problem: $\max_{S \subseteq V} F(S) \text{ s.t. } |S| \leq k$, selecting the most "valuable" events conditioned on the user query.
3.  **Graph-Injected Inference**: The selected events are fed into the LMM with a topology-aware prompt (Topo-CoT).

> **Note**: For the architecture diagram, please refer to Figure 1 in our paper (Asset coming soon).

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/EventGraph.git](https://github.com/yourusername/EventGraph.git)
cd EventGraph

```

### 2. Environment Setup

We recommend using Conda to manage dependencies.

```bash
conda create -n eventgraph python=3.10
conda activate eventgraph

# Install core dependencies
pip install -r requirements.txt

```

### 3. External Dependencies

This project relies on **TransNetV2** for shot detection and **Decord** for video loading.

```bash
pip install transnetv2-pytorch decord

```

---

## 📂 Data Preparation

Please organize your datasets (e.g., VideoMME, CinePile) as follows. The root directory can be configured via arguments.

```text
dataset/
├── VideoMME/
│   ├── videos/       # Contains .mp4 files
│   └── test.json     # Annotation file
├── CinePile/
│   ├── yt_videos/    # Downloaded videos
│   └── ...
└── VRBench/
    └── ...

```

---

## 🏃 Usage

You can run inference using the provided shell script or directly via Python.

### Quick Start

To evaluate on **VideoMME** using **Qwen2.5-VL-7B**:

```bash
bash scripts/run.sh

```

### Advanced Configuration

Run specific configurations via command line parameters:

```bash
python scripts/run_inference.py \
    --dataset VideoMME \
    --data_root ./dataset \
    --backbone Qwen2.5-VL-7B \
    --method EventGraph-LMM \
    --token_budget 8192 \
    --tau 30.0 \
    --delta 0.65

```

#### Key Arguments:

* `--method`: Selection strategy (Default: `EventGraph-LMM`).
* `--tau`: Temporal distance threshold for graph construction (Default: `30.0`).
* `--delta`: Semantic similarity threshold  (Default: `0.65`).
* `--token_budget`: Maximum number of visual tokens allowed.

---

## 📊 Results

**EventGraph-LMM** achieves state-of-the-art performance on VideoMME, CinePile, and VRBench benchmarks.

* **Performance**: Consistently outperforms uniform sampling and other compression baselines.
* **Ablation Findings**: Our experiments demonstrate that  consistently achieves the best performance by effectively filtering irrelevant connections while preserving essential semantic bridges.

*(Detailed results tables can be found in the paper/assets folder)*

---

## 📝 Citation

If you find this project useful, please cite our paper:

```bibtex
@article{eventgraph2026,
  title={EventGraph-LMM: Submodular Subgraph Selection for Token-Efficient Long-Video Understanding},
  author={Anonymous Authors},
  journal={Under Review at ICML},
  year={2026}
}

```