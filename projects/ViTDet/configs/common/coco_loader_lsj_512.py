import detectron2.data.transforms as T
from detectron2 import model_zoo
from detectron2.config import LazyCall as L

# Data using LSJ
image_size = 448
dataloader = model_zoo.get_config("common/data/coco_siz_cropped.py").dataloader
dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True,),  # flip first
    L(T.RandomBrightness)(intensity_min=0.9,intensity_max=1.1),
    L(T.ResizeScale)(
        min_scale=0.5, max_scale=1.5, target_height=image_size, target_width=image_size
    ),
    L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=False),
]
dataloader.train.mapper.image_format = "RGB"
dataloader.train.total_batch_size = 15
# recompute boxes due to cropping
dataloader.train.mapper.recompute_boxes = True

dataloader.test.mapper.augmentations = [
    L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
]
