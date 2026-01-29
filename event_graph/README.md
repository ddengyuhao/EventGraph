EventGraph-LMM: Submodular Subgraph Selection for Token-Efficient Long-Video Understanding<div align="center"></div>Abstract: Understanding long-form videos with Large Multimodal Models (LMMs) is challenging due to the massive token count and limited context windows. We propose EventGraph-LMM, a training-free framework that reformulates long-video compression as a constrained subgraph selection problem. By modeling videos as directed event graphs and applying monotone submodular maximization (using the CELF algorithm), we efficiently select the most informative frames. Furthermore, our Graph-Constrained Chain-of-Thought (Topo-CoT) mechanism guides the reasoning process along verified semantic dependencies.🚀 Key Features📉 Token-Efficient: Reduces visual tokens by 60-80% while maintaining or exceeding the performance of full-context models.🧩 Event Graph Structure: Models video as a graph where nodes are events and edges represent semantic/temporal dependencies ($G=(V, E)$).⚡ Fast Selection: Utilizes the CELF (Cost-Effective Lazy Forward) algorithm for submodular maximization, ensuring a theoretical approximation guarantee with low latency.🧠 Graph-Injected Reasoning: Introduces Topo-CoT to ground the LLM's reasoning path in the constructed graph structure.🔌 Multi-Backbone Support: Seamlessly supports Video-LLaVA, Qwen2-VL, and LLaVA-NeXT.🛠️ FrameworkThe EventGraph PipelineThe framework consists of three stages: Event Graph Construction, Submodular Subgraph Selection, and Graph-Injected Reasoning.Note for Author: Please crop Figure 1 (The main framework diagram) from your PDF and save it as assets/framework.png.Graph Construction: We detect shots (events) and compute semantic similarities to build a directed graph.Subgraph Selection: We solve the submodular maximization problem: $\max_{S \subseteq V} F(S) \text{ s.t. } |S| \leq k$, selecting the most "valuable" events conditioned on the user query.Inference: The selected events are fed into the LMM with a topology-aware prompt.📦 InstallationClone the repositoryBashgit clone https://github.com/yourusername/EventGraph.git
cd EventGraph
Create a Python environmentBashconda create -n eventgraph python=3.10
conda activate eventgraph
Install dependenciesBashpip install -r requirements.txt
Note: Ensure you have ffmpeg installed for video processing.📂 Data PreparationPlease organize your datasets (e.g., VideoMME, CinePile) as follows:Plaintextdataset/
├── VideoMME/
│   ├── videos/
│   └── test.json
├── CinePile/
│   └── ...
🏃 UsageYou can run the inference using the provided shell script or Python command.Quick StartTo evaluate on VideoMME using Qwen2-VL-7B:Bashbash scripts/run.sh
Advanced UsageRun specific configurations via command line:Bashpython scripts/run_inference.py \
    --dataset VideoMME \
    --data_root ./dataset \
    --backbone Qwen2_VL_7B \
    --method eventgraph \
    --frame_budget 128 \
    --graph_lambda 1.0 \
    --sim_threshold 0.65
Key Arguments:--method: Selection strategy (e.g., eventgraph, random, uniform).--graph_lambda: Balance factor for graph connectivity (Default: 1.0).--sim_threshold: Threshold $\delta$ for semantic edges (Default: 0.65).📊 ResultsEventGraph-LMM achieves state-of-the-art performance on VideoMME and VRBench benchmarks.Note for Author: Please crop the Main Results Table (e.g., Table 1 or 2) from your PDF and save it as assets/results.png.Accuracy: Surpasses uniform sampling by X% on VideoMME.Efficiency: Reduces inference cost by significantly lowering the number of input tokens.📝 CitationIf you find this project useful, please cite our paper:代码段@article{eventgraph2026,
  title={EventGraph-LMM: Submodular Subgraph Selection for Token-Efficient Long-Video Understanding},
  author={Anonymous Authors},
  journal={Under Review at ICML},
  year={2026}
}
