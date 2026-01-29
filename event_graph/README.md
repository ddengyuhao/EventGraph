EventGraph-LMM: Submodular Subgraph Selection for Token-Efficient Long-Video Understanding

<div align="center">

</div>

Abstract: Understanding long-form videos with Large Multimodal Models (LMMs) is challenging due to the massive token count and limited context windows. We propose EventGraph-LMM, a training-free framework that reformulates long-video compression as a constrained subgraph selection problem. By modeling videos as directed event graphs and applying monotone submodular maximization (using the CELF algorithm), we efficiently select the most informative frames. Furthermore, we introduce a Graph-Constrained Chain-of-Thought (Topo-CoT) mechanism that guides reasoning along verified semantic dependencies.

🚀 Key Features

📉 Token-Efficient: Reduces visual tokens by 60-80% while maintaining or exceeding the performance of full-context models.

🧩 Event Graph Structure: Models video as a graph $G=(V, E)$ where nodes are events and edges represent semantic/temporal dependencies.

⚡ Fast Selection: Utilizes the CELF (Cost-Effective Lazy Forward) algorithm for submodular maximization, ensuring a theoretical approximation guarantee with low latency.

🧠 Graph-Injected Reasoning: Introduces Topo-CoT to ground the LLM's reasoning path in the constructed graph structure.

🔌 Multi-Backbone Support: Seamlessly supports Video-LLaVA, Qwen2.5-VL, and LLaVA-NeXT.

🛠️ Framework

The EventGraph Pipeline

The framework consists of three stages: Event Graph Construction, Submodular Subgraph Selection, and Graph-Injected Reasoning.

<!--
NOTE FOR AUTHOR:
Please crop the main framework figure (Figure 1 or 2) from 'icml2026 (1).pdf'
and save it as 'https://www.google.com/search?q=assets/framework.png' in your repository.
-->

Graph Construction: We detect shots (events) using TransNetV2 and compute semantic similarities (CLIP) to build a directed graph.

Subgraph Selection: We solve the submodular maximization problem: $\max_{S \subseteq V} F(S) \text{ s.t. } |S| \leq k$, selecting the most "valuable" events conditioned on the user query.

Inference: The selected events are fed into the LMM with a topology-aware prompt (Topo-CoT).

📦 Installation

Clone the repository

git clone [https://github.com/yourusername/EventGraph.git](https://github.com/yourusername/EventGraph.git)
cd EventGraph


Create a Python environment

conda create -n eventgraph python=3.10
conda activate eventgraph


Install dependencies

pip install -r requirements.txt


Note: transnetv2-pytorch and decord are required for video processing.

📂 Data Preparation

Please organize your datasets (e.g., VideoMME, CinePile) as follows. The root directory can be configured in arguments.

dataset/
├── VideoMME/
│   ├── videos/       # Contains .mp4 files
│   └── test.json     # Annotation file
├── CinePile/
│   ├── yt_videos/    # Downloaded videos
│   └── ...
└── VRBench/
    └── ...


🏃 Usage

You can run the inference using the provided shell script or Python command.

Quick Start

To evaluate on VideoMME using Qwen2.5-VL-7B:

bash scripts/run.sh


Advanced Usage

Run specific configurations via command line parameters:

python scripts/run_inference.py \
    --dataset VideoMME \
    --data_root ./dataset \
    --backbone Qwen2.5-VL-7B \
    --method EventGraph-LMM \
    --token_budget 8192 \
    --tau 30.0 \
    --delta 0.65


Key Arguments:

--method: Selection strategy (Default: EventGraph-LMM).

--tau: Temporal distance threshold for graph construction (Default: 30.0).

--delta: Semantic similarity threshold $\delta$ (Default: 0.65, as validated in paper ablation).

--token_budget: Maximum number of visual tokens allowed.

📊 Results

EventGraph-LMM achieves state-of-the-art performance on VideoMME, CinePile, and VRBench benchmarks.

<!--
NOTE FOR AUTHOR:
Please crop the Main Results Table (e.g., Table 1) from 'icml2026 (1).pdf'
and save it as 'https://www.google.com/search?q=assets/results.png'.
-->

Performance: Consistently outperforms uniform sampling and other compression baselines.

Ablation Findings: Our experiments demonstrate that $\delta=0.65$ consistently achieves the best performance, effectively filtering out irrelevant connections while preserving essential semantic bridges.

📝 Citation

If you find this project useful, please cite our paper:

@article{eventgraph2026,
  title={EventGraph-LMM: Submodular Subgraph Selection for Token-Efficient Long-Video Understanding},
  author={Anonymous Authors},
  journal={Under Review at ICML},
  year={2026}
}

