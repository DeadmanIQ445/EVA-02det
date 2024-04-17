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

import slidingwindow
from detectron2.config import LazyConfig, instantiate

cfg = LazyConfig.load("/home/arina/repos/EVA/EVA-02/det/projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_b_4attn_2048_lrd0p7.py")

# default_setup(cfg, )

out_dir = 'pred_bpla_out_2048_2'
model = instantiate(cfg.model)
model.to(cfg.train.device)
model = create_ddp_model(model)




DetectionCheckpointer(model).load('/home/arina/repos/EVA/EVA-02/det/bpla_res_0_2/model_0001999.pth')  
os.makedirs(out_dir, exist_ok=True)
model.eval()
for i in glob.glob('/home/arina/projs/spils/BPLA/reannotated/yolo_01_drone_081223/images/val/*'):
    img = cv2.imread(i)
    height, width = img.shape[:2]
    windows = slidingwindow.generate(img, slidingwindow.DimOrder.HeightWidthChannel, 2048, 0.5)
    factor = max(height, width)/2048
    dim = (int(height//factor), int(width//factor))
    # print(dim, type(dim[1]))
    img2 = cv2.resize(img, (dim[1], dim[0]))
    with torch.no_grad():

        height, width = dim

        image = torch.as_tensor(img2.astype("float32").transpose(2, 0, 1), device='cuda')

        inputs = {"image": image, "height": torch.tensor(height), "width": torch.tensor(width)}
        y = model.forward([inputs])
        y_cpu = y[0]['instances'].to('cpu')
    masks = y_cpu.pred_masks.numpy().astype("uint8")
    classes = y_cpu.pred_classes.numpy()
    scores = y_cpu.scores.numpy()
    print(i)
    for j in zip(classes, y_cpu.scores.numpy(),y_cpu.pred_boxes):
        print(j)

    for mask, cls, score in zip(masks, classes, scores):
        if score<0.1: 
            continue
        color = (0,0,255) if cls==0 else (0,255,0)
        mask = np.expand_dims(mask, axis=2)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            cv2.polylines(img,contours,True,color,10)
    cv2.imwrite(out_dir+'/'+os.path.basename(i), img)