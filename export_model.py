from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import torch
import glob
import cv2
import numpy as np
import os
from rasterio.plot import reshape_as_raster 
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model


from detectron2.config import LazyConfig, instantiate

cfg = LazyConfig.load("/home/arina/repos/EVA/EVA-02/det/projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_b_4attn_2048_lrd0p7.py")

# default_setup(cfg, )

out_dir = 'pred_bpla_winter_2048_3'
model = instantiate(cfg.model)
model.to(cfg.train.device)
# model = create_ddp_model(model)




DetectionCheckpointer(model).load('/home/arina/repos/EVA/EVA-02/det/outputs/bpla_winter_oil_2048/model_0001999.pth')  
model.eval()

for i in glob.glob('/home/arina/projs/spils/BPLA_winter/yolo_01/images/val/*'):
    img = cv2.imread(i)
    height, width = img.shape[:2]
    factor = max(height, width)/2048
    dim = (int(height//factor), int(width//factor))
    # print(dim, type(dim[1]))
    img2 = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (dim[1], dim[0]))
    height, width = dim
    image = torch.as_tensor(img2.astype("float32").transpose(2, 0, 1), device='cuda')
    inputs = {"image": image, "height":
     torch.tensor(height), "width": torch.tensor(width)}
    print(len(inputs['image']))
    torch.onnx.export(model,[inputs], 'export.onnx', )
    break
    # y = model.forward([inputs])