# 注册表

# EventGraph-LMM实现（Ours - ICML 2026）
from .eventgraph import EventGraphLMM

# No-Compression Baseline（诊断用基准）
from .no_compression import NoCompression

METHOD_REGISTRY = {
    "EventGraph-LMM": EventGraphLMM,  # 主方法
    "No-Compression": NoCompression  # 诊断基准
}