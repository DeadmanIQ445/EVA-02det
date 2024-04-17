
import torch
import glob
import cv2
import numpy as np
import os


# default_setup(cfg, )

out_dir = 'pred_bpla_winter_test_ts'
model = torch.jit.load('out/model.ts')




os.makedirs(out_dir, exist_ok=True)
model.eval()

for i in glob.glob('/home/arina/projs/spils/BPLA_winter/yolo_01/images/val/*'):
    img = cv2.imread(i)
    height, width = img.shape[:2]
    factor = max(height, width)/2048
    dim = (int(height//factor), int(width//factor))
    # print(dim, type(dim[1]))
    img2 = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (dim[1], dim[0]))
    with torch.no_grad():

        height, width = dim

        image = torch.as_tensor(img2.astype("float32").transpose(2, 0, 1), device='cuda')

        inputs = {"image": image, "height": torch.tensor(height), "width": torch.tensor(width)}
        y = model.forward(inputs['image'])
        # print(y)
    masks = y[2].cpu().numpy()
    classes = y[1].cpu().numpy()
    scores = y[3].cpu().numpy()
    # print(i)
    # for j in zip(classes, y_cpu.scores.numpy(),y_cpu.pred_boxes):
    #     print(j)

    for mask, cls, score in zip(masks, classes, scores):
        if score<0.1: 
            continue
        color = (0,0,255) if cls==0 else (0,255,0)

        mask = np.expand_dims(mask.squeeze().astype(np.uint8), axis=2)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            cv2.polylines(img,contours,True,color,10)
    cv2.imwrite(out_dir+'/'+os.path.basename(i), img)