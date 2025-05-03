import torch
import clip
from baseline import yolo_model, clip_model
from util import calculate_IoU, compare_bbox, draw_bbox
from model import VisualLanguisticTranformer
from tqdm import tqdm
import numpy as np
from types import SimpleNamespace

device = "cuda" if torch.cuda.is_available() else "cpu"

def eval_loop(data):
    correct = 0
    total = 0
    iou_array = []
    model, preprocess = clip.load("RN50", device=device)
    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in tqdm(data, desc="Processing Test Data"):
            cropped_images, bboxes = yolo_model(yolo, sample['image'])

            pred_bbox = clip_model(model, preprocess, cropped_images, bboxes, sample['description'])

            iou = calculate_IoU(pred_bbox, sample['bbox'])

            iou_array.append(iou)

            if  iou >= 0.5:
                correct += 1
            total += 1

    return correct/total, np.asarray(iou_array).mean()


def model_test(data):
    model, preprocess = clip.load("RN50", device="cpu")

    vg = VisualLanguisticTranformer(model)

    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in tqdm(data, desc="Processing Test Data"):
            tokens = clip.tokenize(sample['description'])
            image = preprocess(sample['image']).unsqueeze(0)
            vg(image, tokens)
