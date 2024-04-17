from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import torch


from detectron2.engine.defaults import create_ddp_model


from detectron2.config import LazyConfig, instantiate

cfg = LazyConfig.load("/home/arina/repos/EVA/EVA-02/det/projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_b_4attn_2048_lrd0p7.py")

# default_setup(cfg, )

out_dir = 'pred_bpla_winter_2048_3'
model = instantiate(cfg.model)
model.to(cfg.train.device)
model = create_ddp_model(model)




DetectionCheckpointer(model).load('/home/arina/repos/EVA/EVA-02/det/outputs/bpla_winter_oil_2048/model_0001999.pth')  
model.eval()
torch.save()