from .videomme import VideoMME
from .longvideobench import LongVideoBench
from .cinepile import CinePileDataset
from .egoschema import EgoSchemaDataset
# [新增]
from .mlvu import MLVUDataset

DATASET_REGISTRY = {
    "VideoMME": VideoMME,
    "LongVideoBench": LongVideoBench,
    "CinePile": CinePileDataset,
    "EgoSchema": EgoSchemaDataset,
    "MLVU": MLVUDataset, 
}