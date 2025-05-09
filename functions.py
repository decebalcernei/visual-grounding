import torch
import clip
from baseline import yolo_model, clip_model
from util import calculate_IoU, compare_bbox, draw_bbox
from model import VisualLanguisticTranformer
from tqdm import tqdm
import numpy as np
from types import SimpleNamespace

device = "cuda" if torch.cuda.is_available() else "cpu"

def eval_loop_baseline(data):
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


def train_loop(model, data, optimizer, device):
    model.train()
    loss_array = []
    for sample in tqdm(data, desc="Processing Training Dataset"):
        images = sample["image"].to(device)
        descriptions = sample["description"].to(device)
        gt_bboxes = sample["bbox"].to(device)

        predicted_bboxes = model(images, descriptions)
        
        iou = calculate_IoU(gt_bboxes, predicted_bboxes)
        loss = 1 - iou.mean()  # we wanna maximize the iou so we minimize 1-iou
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # Update the weights

        loss_array.append(loss.item())
    
    return loss_array



def eval_loop(model, dataloader, device):
    model.eval()  # metti il modello in modalità evaluation
    all_ious = []

    with torch.no_grad():  # disabilita il tracking del gradiente
        for sample in tqdm(dataloader, desc="Evaluating"):
            images = sample["image"].to(device)
            descriptions = sample["description"].to(device)
            gt_bboxes = sample["bbox"].to(device)

            # Predici le bbox
            predicted_bboxes = model(images, descriptions)

            # Calcola la IoU per ogni coppia predetta/ground-truth
            ious = calculate_IoU(gt_bboxes, predicted_bboxes)

            all_ious.extend(ious.tolist())  # aggiungili alla lista globale

    # Calcola e ritorna la IoU media
    mean_iou = sum(all_ious) / len(all_ious)
    return mean_iou
