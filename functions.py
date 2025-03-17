import torch
import clip
from baseline import yolo_model, clip_model
from util import calculate_IoU, compare_bbox, draw_bbox

device = "cuda" if torch.cuda.is_available() else "cpu"

def eval_loop(data):
    correct = 0
    total = 0
    model, preprocess = clip.load("RN50", device=device)
    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    cropped_images, bboxes = yolo_model(yolo, data[186]['image'])
    
    
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            cropped_images, bboxes = yolo_model(yolo, sample['image'])

            pred_bbox = clip_model(model, preprocess, cropped_images, bboxes, sample['description'])

            iou = calculate_IoU(pred_bbox, sample['bbox'])
            if  iou >= 0.5:
                correct += 1
            total += 1

    return correct/total
