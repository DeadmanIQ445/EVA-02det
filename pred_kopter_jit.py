import torch
import glob
import cv2
import numpy as np
import os
import copy
import pandas as pd

import torch
from torch.nn import functional as F

from detectron2.structures import Instances, ROIMasks, Boxes


import shapely
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd
# perhaps should rename to "resize_instance"
def detector_postprocess(
    results: Instances, output_height: int, output_width: int, mask_threshold: float = 0.5
):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.
    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    if isinstance(output_width, torch.Tensor):
        # This shape might (but not necessarily) be tensors during tracing.
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height
    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    print(scale_x, scale_y)
    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)
    # results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        if isinstance(results.pred_masks, ROIMasks):
            roi_masks = results.pred_masks
        else:
            # pred_masks is a tensor of shape (N, 1, M, M)
            roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
        results.pred_masks = roi_masks.to_bitmasks(
            results.pred_boxes, output_height, output_width, mask_threshold
        ).tensor  # TODO return ROIMasks/BitMask object in the future

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results


def extract_polygons(mask):
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    polygons = []
    for cnt in contours:
        if cnt.shape[0]<3:
            continue
        pol = Polygon(np.squeeze(cnt))
        if pol.area<50:
            continue
        if not pol.is_valid:
            pol = pol.buffer(0)
        if isinstance(pol, MultiPolygon):
            for pol_1 in list(pol.geoms):
                polygons.append(pol_1)
        else:
            polygons.append(pol)
    return polygons

def _iou_( test_poly, truth_poly):
    """Intersection over union"""
    try:
        test_poly, truth_poly = Polygon(test_poly), Polygon(truth_poly)
        intersection_result = test_poly.intersection(truth_poly)
        intersection_area = intersection_result.area
        union_area = test_poly.union(truth_poly).area
        return (intersection_area / union_area), \
                intersection_area / test_poly.area, \
                intersection_area / truth_poly.area

    except shapely.topology.TopologicalError:
        return 0, 0, 0


def join_nms(mosaic_df: pd.DataFrame, iou_threshold, corr_coef):
    """
        Non-max supression for overlapping boxes among window

        Args:
            mosaic_df (): array of predicted instances
            iou_threshold (): overlap threshold
            corr_coef (): correlation coefficient

        Returns: list of boxes after NMS

    """
    ret_boxes = []
    boxes_cp = copy.deepcopy(mosaic_df.to_dict('records'))

    while len(boxes_cp) > 0:
        m = min(range(len(boxes_cp)), key=lambda i: boxes_cp[i]['area'])

        b_m = boxes_cp[m]
        boxes_cp.pop(m)
        flag = True
        for i in boxes_cp:
            poly_m = b_m['geometry']
            poly_i = i['geometry']
            iou, poly_m_inter, poly_i_inter = _iou_(poly_m, poly_i)
            # if iou > 0:
            #     iou = iou
            if poly_m_inter > corr_coef:
                flag = False
                break
            if poly_i_inter > corr_coef:
                # if i['score']-b_m['score']>0.5:
                #     flag=False
                #     break
                continue
            if iou > iou_threshold:
                if b_m['score'] < i['score']:
                    flag = False
                    break
        if flag:
            ret_boxes.append(b_m)
    return ret_boxes


name = 'cropped_250'

model_path = f'/home/ibragim/repos/EVA-02det/jit/siz/{name}/model.ts'

threshold = 0.5
out_dir = f'predictions/siz/{name}'

if torch.cuda.is_available():
    device='cuda'
else:
    device = 'cpu'

model = torch.jit.load(model_path, device)
model.to(device)
os.makedirs(out_dir, exist_ok=True)
# for i in glob.glob('/home/arina/projs/spils/kopter_summer/drone_5.1_filtered_06082024/12082024_w_bpla/images/test/*'):
# for i in glob.glob('/home/arina/projs/karyeri/copter_summer_2.0/yolo_0_05092024/images/test/*'):
#
for i in glob.glob('/home/ibragim/data/siz_segment/cropped_20241001/images/test/*'):
    img = cv2.imread(i)
    height, width = img.shape[:2]

    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img2 = cv2.resize(img2, (dim[1], dim[0]))
    img2 = cv2.resize(img2, (250, 250))
    # img2 = cv2
    with torch.no_grad():
        # image = torch.as_tensor(img2.astype("float32").transpose(2, 0, 1), device='cuda')

        # inputs = {"image": image, "height": torch.tensor(height), "width": torch.tensor(width)}
        inputs = torch.tensor(img2.transpose(2, 0, 1).astype(np.float32), device=device)
        print(inputs.shape)
        with torch.jit.optimized_execution(False):
            ort_outs = model(inputs)
        # print(ort_outs)

        y = ort_outs
        instances= Instances((ort_outs[-1][0].cpu().numpy(),ort_outs[-1][1].cpu().numpy()))
        instances.set('pred_boxes', Boxes(ort_outs[0].cpu().numpy()))
        instances.set('pred_classes', ort_outs[1].cpu().numpy())
        instances.set('pred_masks', torch.tensor(ort_outs[2].cpu().numpy()))
        instances.set('scores', ort_outs[3].cpu().numpy())
        postprocessed = detector_postprocess(instances,img.shape[0], img.shape[1])
        y_cpu = postprocessed
        # print(postprocessed)
    masks = y_cpu.pred_masks.cpu().numpy().astype("uint8")
    classes = y_cpu.pred_classes
    scores = y_cpu.scores

    print(i)
    for j in zip(classes, scores,y_cpu.pred_boxes):
        print(j)




    os.makedirs(out_dir+"_gpd_nms", exist_ok=True)

    res = []
    output_mask = np.zeros((img.shape[0], img.shape[1],1))
    for mask, cls, score in zip(masks, classes, scores):
        if score<threshold: 
            continue
        color = 1 if cls==0 else 2
        mask = np.expand_dims(mask, axis=2)
        # mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        polys = extract_polygons(mask)
        for poly in polys:
            res.append({'score':score, 'class':cls, 'geometry': poly, 'area':poly.area })

    res = gpd.GeoDataFrame(res)
    # res = gpd.GeoDataFrame(join_nms(res,1.0,1.0))
    colors = [(255,0,0), (0,255,0), (0,0,255), (255, 255, 0), (0, 255, 255), (255, 0, 255) ]
    classes = {0: "Hardhat", 1: "No-Hardhat",2: "Safety Jacket",3: "NO-Safety Jacket",4: "Safety Pants",5: "NO-Safety Pants",6: "BagGasMask"}
    if len(res)>0:
        os.makedirs(out_dir, exist_ok=True)
        img = cv2.imread(i)
        for idx,pol in res.iterrows():
            if pol['class']<1:
                continue
            color =  colors[pol['class']-1]
            cv2.drawContours(img, [np.array(pol['geometry'].exterior.coords, dtype=int)], -1, color, 5)
            x,y,x2,y2 = pol['geometry'].bounds
            img=cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), color, 1)
            cv2.putText(img, classes[pol['class']-1], (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        res.to_file(out_dir+'_gpd_nms/'+os.path.basename(i)[:-4]+'.geojson', driver='GeoJSON')
    else:
        gpd.GeoDataFrame({'class':[0], 'score':[0], 'geometry':[Polygon([(0,0),(0,0), (0,0), (0,0)])]}).to_file(out_dir+'_gpd_nms/'+os.path.basename(i)[:-4]+'.geojson', driver='GeoJSON')
    cv2.imwrite(os.path.join(out_dir, os.path.basename(i)), img)
        
        
        
        