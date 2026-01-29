# 注册表

# EventGraph-LMM实现（Ours - ICML 2026）
from .eventgraph import EventGraphLMM

# No-Compression Baseline（诊断用基准）
from .no_compression import BaselineUniform

METHOD_REGISTRY = {
    "EventGraph-LMM": EventGraphLMM,  # 主方法
    "No-Compression": BaselineUniform  # 诊断基准
}