from .cinepile import CinePileDataset
from .vrbench import VRBenchDataset
from .videomme import VideoMMEDataset
# 1. 导入新类
from .longvideobench import LongVideoBenchDataset
from .lvbench import LVBenchDataset

DATASET_REGISTRY = {
    "CinePile": CinePileDataset,
    "VRBench": VRBenchDataset,
    "VideoMME": VideoMMEDataset,
    # 2. 注册名字
    "LongVideoBench": LongVideoBenchDataset,
    "LVBench": LVBenchDataset
}