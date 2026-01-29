from .cinepile import CinePileDataset
from .vrbench import VRBenchDataset
from .videomme import VideoMMEDataset

DATASET_REGISTRY = {
    "CinePile": CinePileDataset,
    "VRBench": VRBenchDataset,
    "VideoMME": VideoMMEDataset,
}