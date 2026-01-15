# /root/ICML2026/event_graph/my_dataset.py

from .base_dataset import BaseDataset
from .cinepile import CinePileDataset
# 1. 导入新类
from .vrbench import VRBenchDataset 

DATASET_REGISTRY = {
    "CinePile": CinePileDataset,
    # 2. 注册名字
    "VRBench": VRBenchDataset
}