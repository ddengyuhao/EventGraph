from .videomme import VideoMMEDataset
from .longvideobench import LongVideoBenchDataset
from .cinepile import CinePileDataset

DATASET_REGISTRY = {
    "VideoMME": VideoMMEDataset,
    "LongVideoBench": LongVideoBenchDataset,
    "CinePile": CinePileDataset,
    # "EgoSchema": EgoSchemaDataset,
    # "MLVU": MLVUDataset, 
}