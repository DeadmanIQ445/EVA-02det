import cv2
import numpy as np
from random import randint
import glob
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score
from shapely.geometry import Polygon, MultiPolygon
import tqdm
import geopandas as gpd

def cvt_to_mask(img_path, labels_file):
    with open(labels_file, 'r') as f:
        labels = f.read().splitlines()
    img = np.zeros_like(cv2.imread(img_path))
    h,w = img.shape[:2]

    for label in labels:
    #             print(line)
        strip = label.strip().split()
        coords = strip[1:]
        coords = np.array(list(map(float, coords)))
        coords = coords.reshape(-1,2)
    #             print(coords)
        h,w,c = img.shape
        coords[:,0]*=w
        coords[:,1]*=h
        coords = coords.reshape(-1,1,2).astype(np.int32).squeeze()
        img = cv2.drawContours(img,[coords], -1,color=(int(strip[0])+1,int(strip[0])+1,int(strip[0])+1), thickness=-1)
    # return img
    return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

IOU_metrics_thresh = 0.5

def calc_IoU(a,b):
    if isinstance(a, Polygon):
        return a.intersection(b).area/a.union(b).area
    else:
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        
        minx = max(a[0], b[0])
        maxx = min(a[2], b[2])
        miny = max(a[1], b[1])
        maxy = min(a[3], b[3])
        
        intersection = max(0, maxx-minx) * max(0,maxy-miny)
        union = area_a + area_b - intersection
        return intersection/union

def calculate_metrics(pred_boxes, gt_boxes):
    TP = 0
    FP = 0
    if len(pred_boxes)==0:
        if len(gt_boxes)==0:
            return 1,1,1
        else:
            return 0,0,0
    if len(gt_boxes)==0:
        if len(pred_boxes)>0:
            return 0,1,0
    gt_false = gt_boxes.copy()
    for pred in pred_boxes:
        flag = False
        for gt_i in range(len(gt_false)):
            temp_IoU = calc_IoU(pred,gt_false[gt_i])
            if temp_IoU>IOU_metrics_thresh:
                TP+=1
                del gt_false[gt_i]
                flag = True
                break
        if not flag:
            FP+=1
    FN = len(gt_false)
    
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    if Precision+Recall==0:
        F1= 0
    else:
        F1 = (Precision * Recall)/((Precision + Recall)/2)
    return Precision, Recall, F1
                        
def extract_boxes(mask):
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    boxes = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        boxes.append((x,y,x+w, y+h))
    return boxes

 
def extract_polygons(mask):
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_L1)
    polygons = []
    for cnt in contours:
        if len(cnt)<3:
            continue
        pol = Polygon(np.squeeze(cnt))
        if pol.area<100:
            continue
        if not pol.is_valid:
            pol = pol.buffer(0)
        if isinstance(pol, MultiPolygon):
            for pol_1 in list(pol.geoms):
                if pol_1.area>100:
                    polygons.append(pol_1)
        else:
            polygons.append(pol)
    return polygons

def calculate_multiclass(pred, gt, n_classes=None):
    classes = set(np.unique(pred['class'])+1).union(set(np.unique(gt)))
    if not n_classes: n_classes = max(classes)
    f1s = {}
    rs = {}
    ps = {}

    for i in classes:
        if i==0: continue
        pred_i = pred[pred['class']==i-1]
        gt_i = gt.copy()
        gt_i[gt_i!=i] = 0
        P,R,F1 = calculate_metrics(pred_i.geometry, extract_polygons(gt_i))
        f1s[i]=F1
        rs[i]=R
        ps[i]=P
    print(f1s,rs,ps)
    return ps,rs,f1s
    

def calculate_directory(pred_folder, image_folder,class_agnostic=False, score_thresh=None):
    f1s_multiclass = {}
    ps_multiclass={}
    rs_multiclass={}
    ps_per_image = []
    rs_per_image=[]
    f1s_per_image=[]
    files = []
    tqdm_for = tqdm.tqdm(os.listdir(pred_folder))
    for file in tqdm_for:
        tqdm_for.set_description(file)
        img_path = os.path.join(image_folder,file.rsplit('.',1)[0]+'.JPG')
        if not os.path.exists(img_path):
            img_path = img_path.replace('.JPG','.png')
        if not os.path.exists(img_path):
            img_path = img_path.replace('.png','.jpg')
        labels_path = img_path.replace('images','labels')[:-4]+'.txt'

        gt = cvt_to_mask(img_path,labels_path)
        pred =gpd.read_file(os.path.join(pred_folder,file))
        if score_thresh:
            pred = pred[pred['score']>score_thresh]
        if class_agnostic: 
            gt[gt>0]=1
            pred['class']=0
        P,R,F1 = calculate_multiclass(pred,gt)
        files.append(file)
        for i in F1.keys():
            if i in f1s_multiclass:
                f1s_multiclass[i].append(F1[i])
            else:
                f1s_multiclass[i] = [F1[i]]
            if i in ps_multiclass:
                ps_multiclass[i].append(P[i])
            else:
                ps_multiclass[i] = [P[i]]
            if i in rs_multiclass:
                rs_multiclass[i].append(R[i])
            else:
                rs_multiclass[i] = [R[i]]

        rs_per_image.append(np.mean(list(R.values())))
        ps_per_image.append(np.mean(list(P.values())))

        f1s_per_image.append(np.mean(list(F1.values())))

    return files, ps_multiclass, rs_multiclass, f1s_multiclass, ps_per_image, rs_per_image, f1s_per_image


def calculate_image(pred_image, img_path, class_agnostic=False):
    labels_path = img_path.replace('images','labels')[:-4]+'.txt'

    gt = cvt_to_mask(img_path,labels_path)
    pred =cv2.cvtColor(cv2.imread(pred_image),cv2.COLOR_RGB2GRAY)
    if class_agnostic: 
        gt[gt>0]=1
        pred[pred>0]=1
    P,R,F1 = calculate_multiclass(pred,gt)
    return P,R,F1


if __name__=='__main__':
    # files, ps_multiclass, rs_multiclass, f1s_multiclass,\
    # ps_per_image, rs_per_image, f1s_per_image = calculate_directory('/home/arina/repos/EVA/EVA-02/det/predictions_ts_2008_3/oil/1536_b_w_bpla_gpd_nms',
    #                                     '/home/arina/projs/spils/kopter_summer/drone_5.1_filtered_06082024/12082024_w_bpla/images/test',
    #                                     class_agnostic=False, score_thresh=0.35)
    files, ps_multiclass, rs_multiclass, f1s_multiclass,\
    ps_per_image, rs_per_image, f1s_per_image = calculate_directory('/home/ibragim/repos/EVA-02det/predictions/siz/cropped_640_gpd_nms',
                                    '/home/ibragim/data/siz_segment/cropped_20241001/images/test/',
                                    class_agnostic=False, score_thresh=0.1)

    print(np.mean(f1s_per_image), np.mean(rs_per_image), np.mean(ps_per_image))
    for i in range(len(f1s_multiclass)):
        print('Class:',i+1, (np.mean(f1s_multiclass[i+1]), np.mean(rs_multiclass[i+1]), np.mean(ps_multiclass[i+1])))
    # print(np.mean(f1s_multiclass[2]), np.mean(rs_multiclass[2]), np.mean(ps_multiclass[2]))
    with open('toCheck.txt', 'w') as f:
        for i in sorted(zip(f1s_per_image, rs_per_image, ps_per_image,files)):
            if f1s_per_image != 1:
                print(i[3], i[0],i[1],i[2])
                f.write(f'{i[3]}, {i[0]},{i[1]},{i[2]}\n')