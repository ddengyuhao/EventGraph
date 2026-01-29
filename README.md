# EventGraph-LMM: Submodular Subgraph Selection for Token-Efficient Long-Video Understanding

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10-green.svg)]()
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)]()
[![Paper](https://img.shields.io/badge/Paper-ICML%202026-lightgrey)]()

**EventGraph-LMM** is a **training-free** framework for long-video understanding. It reformulates video compression as a **constrained subgraph selection problem**, achieving SOTA performance on VideoMME, VRBench, and CinePile with minimal token usage.

[<a href="#-news">News</a>] • [<a href="#-abstract">Abstract</a>] • [<a href="#-framework">Framework</a>] • [<a href="#-performance">Performance</a>] • [<a href="#-installation">Installation</a>] • [<a href="#-usage">Usage</a>] • [<a href="#-citation">Citation</a>]

</div>

## 📖 Abstract

Understanding long-form videos with Large Multimodal Models (LMMs) is challenging due to massive visual token counts and limited context budgets. We propose **EventGraph-LMM**, a training-free framework that reformulates long-video compression as a **constrained subgraph selection problem** rather than simple token pruning.

By modeling videos as directed event graphs that capture both temporal flow and long-range semantic dependencies, we formulate subgraph selection as a **monotone submodular maximization problem**. This allows us to use the **CELF algorithm** to efficiently select the most informative frames with theoretical approximation guarantees. Furthermore, we introduce a **Graph-Constrained Chain-of-Thought (Graph-CoT)** mechanism that guides the LMM's reasoning along verified visual dependencies, significantly reducing hallucinations.

---

## 🛠️ Framework

> **Figure 1**: The overall framework of EventGraph-LMM. We first decompose the video into events, construct a weighted graph capturing semantic and temporal links, and then select a budget-constrained subgraph to guide the LMM's reasoning.

<div align="center">
  <img src="assets/framework.png" width="95%" alt="EventGraph Framework"/>
  <br>
</div>

### Key Components

1.  **Event Graph Construction**: 
    * Decomposes video into events using **TransNetV2**.
    * Extracts global ([CLS]) and local patch features using **CLIP**.
    * Builds a graph $G=(V, E)$ with **temporal edges** (sequential flow) and **semantic edges** (long-range similarity).
2.  **Constrained Subgraph Selection**: 
    * Solves the submodular optimization problem.
    * The objective function balances **Query Relevance** and **Reachable Information Gain**.
    * Uses the **CELF algorithm** for fast, near-optimal selection.
3.  **Graph-Constrained Reasoning (Graph-CoT)**:
    * Guides the LMM to verify evidence and propagate logic strictly along the selected graph paths.

---

## 📊 Performance

We evaluate EventGraph-LMM on **VideoMME**, **VRBench**, and **CinePile**. Under a strict budget of **8,192 visual tokens**, our method significantly outperforms existing efficient inference baselines.

| Method | Type | LLM Backbone | VideoMME | VRBench | CinePile | Avg. |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| *Proprietary LMMs* | | | | | | |
| Gemini 1.5 Pro | API | - | 75.0 | 70.7 | 60.1 | 68.6 |
| GPT-4o | API | - | 71.9 | 68.7 | 56.1 | 65.6 |
| *Open-Source LMMs (Full-Sequence)* | | | | | | |
| InternVL2.5-72B | Dense | Qwen2.5-72B | 72.1 | 53.5 | 54.6 | 60.1 |
| Qwen2-VL-72B | Dense | Qwen2-72B | 71.2 | 59.1 | 54.2 | 61.5 |
| Qwen2.5-VL-7B | Dense | Qwen2.5-7B | 65.1 | 56.5 | 52.6 | 58.1 |
| LLaVA-NeXT-34B | Dense | Qwen1.5-34B | 70.6 | 48.5 | 41.5 | 53.5 |
| *Efficient Strategies (Qwen2.5-VL-7B)* | | | | | | |
| LLaVA-Phi | Architecture | Phi-2 | 34.5 | 22.0 | 18.5 | 25.0 |
| FastV | Token Reduction | Qwen2.5-VL-7B | 52.3 | 43.1 | 38.4 | 44.6 |
| DyCoke | Token Reduction | Qwen2.5-VL-7B | 51.8 | 47.2 | 37.1 | 45.4 |
| Q-Frame | Keyframe Sampling | Qwen2.5-VL-7B | 53.5 | 48.5 | 40.7 | 47.6 |
| MovieChat | Memory | Qwen2.5-VL-7B | 48.5 | 35.0 | 28.0 | 37.2 |
| SGVC | Caption | Qwen2.5-VL-7B | 45.0 | 32.0 | 30.2 | 35.7 |
| **Ours** | **Graph** | **Qwen2.5-VL-7B** | **58.5** | **54.8** | **48.1** | **53.8** |
| *Efficient Strategies (Qwen2-VL-72B)* | | | | | | |
| FastV | Token Reduction | Qwen2-VL-72B | 56.5 | 45.5 | 35.2 | 45.7 |
| DyCoke | Token Reduction | Qwen2-VL-72B | 57.9 | 46.8 | 34.5 | 46.4 |
| Q-Frame | Keyframe Sampling | Qwen2-VL-72B | 62.0 | 52.1 | 39.5 | 52.2 |
| **Ours** | **Graph** | **Qwen2-VL-72B** | **66.2** | **58.5** | **51.3** | **58.7** |


> *Results sourced from Table 1 in our paper.*

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
conda create -n eventgraph python=3.10 -y
conda activate eventgraph

# Install core dependencies
pip install -r requirements.txt

# Install TransNetV2 (for shot detection) and Decord (for video loading)
pip install transnetv2-pytorch decord

```

---

## 📂 Data Preparation

Please organize your datasets as follows. You can configure the `DATA_ROOT` in `scripts/run.sh`.

```text
dataset/
├── VideoMME/
│   ├── videos/       # Contains .mp4 files
│   └── test.json     # Annotation file (Hardcoded in code)
├── CinePile/
│   ├── yt_videos/    # Downloaded video files
│   └── cookies.txt   # (Optional) YouTube auth for downloading restricted videos
└── VRBench/
    ├── videos/       # Contains video files
    └── VRBench.json  # Annotation file
```

---

## 🏃 Usage

You can run inference using the provided shell script, which supports multi-GPU chunking.

### Quick Start

To evaluate on **VideoMME** using **Qwen2.5-VL-7B**:

```bash
bash scripts/run.sh

```

### Advanced Configuration

You can also run the python script directly for specific configurations:

```bash
python scripts/run_inference.py \
    --dataset VideoMME \
    --data_root ./dataset \
    --backbone Qwen2.5-VL-7B \
    --method EventGraph-LMM \
    --token_budget 8192 \
    --tau 30.0 \
    --delta 0.65 \
    --output_dir ./results/debug

```

#### Key Arguments:

* `--method`: Selection strategy (Default: `EventGraph-LMM`).
* `--backbone`: Model backbone (e.g., `Qwen2.5-VL-7B`, `Qwen2-VL-72B`).
* `--token_budget`: Maximum number of visual tokens allowed (default: 8192).
* `--tau`: Temporal distance threshold for graph construction (default: 30.0).
* `--delta`: Semantic similarity threshold (default: 0.65).



---

## 📝 Citation

If you find this project useful for your research, please cite our paper:

```bibtex
@article{eventgraph2026,
  title={EventGraph-LMM: Submodular Subgraph Selection for Token-Efficient Long-Video Understanding},
  author={Anonymous Authors},
  journal={Under Review at ICML},
  year={2026}
}

```