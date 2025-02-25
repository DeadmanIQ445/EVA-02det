from functools import partial

from ..common.coco_loader_lsj_2048 import dataloader
from .cascade_mask_rcnn_vitdet_b_100ep import (
    # dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)

train.init_checkpoint = "/home/ibragim/repos/EVA-02det/eva02_L_coco_sys.pth"

model.backbone.net.img_size = 2048  
model.backbone.square_pad = 2048  
model.backbone.net.patch_size = 16  
model.backbone.net.window_size = 32
model.backbone.net.embed_dim = 1024
model.backbone.net.depth = 24
model.backbone.net.num_heads = 16
model.backbone.net.mlp_ratio = 4*2/3
model.backbone.net.use_act_checkpoint = True
model.backbone.net.drop_path_rate = 0.4  

# 2, 5, 8, 11, 14, 17, 20, 23 for global attention
model.backbone.net.window_block_indexes = (
    list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11)) + list(range(12, 14)) + list(range(15, 17)) + list(range(18, 20)) + list(range(21, 23))
)

optimizer.lr=5e-6
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, lr_decay_rate=0.8, num_layers=24)
optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None

model.roi_heads.num_classes = 7


train.max_iter = 20000
lr_multiplier.scheduler.milestones = [
    train.max_iter*2//10, train.max_iter*9//10
]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 500 / train.max_iter

dataloader.test.num_workers=0
dataloader.train.total_batch_size=6

