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
train.init_checkpoint = "/home/arina/repos/EVA/EVA-02/det/eva02_B_coco_bsl.pth"

model.backbone.net.img_size = 2048 
model.backbone.square_pad = 2048  
model.backbone.net.patch_size = 16  
model.backbone.net.window_size = 32 
model.backbone.net.embed_dim = 768
model.backbone.net.depth = 12
model.backbone.net.num_heads = 12
model.backbone.net.mlp_ratio = 4*2/3
model.backbone.net.use_act_checkpoint = False
model.backbone.net.drop_path_rate = 0.1
# 2, 5, 8, 11 for global attention
model.backbone.net.window_block_indexes = [0, 1, 3, 4, 6, 7, 9, 10]

# for i in model.backbone.layers:
#     i.requires_grad = False
model.roi_heads.num_classes = 2

optimizer.lr=5e-6
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, lr_decay_rate=0.7, num_layers=12)
optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None


train.max_iter = 60000
lr_multiplier.scheduler.milestones = [
    train.max_iter*8//10, train.max_iter*9//10
]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 1000 / train.max_iter

dataloader.test.num_workers=0
dataloader.train.total_batch_size=6

