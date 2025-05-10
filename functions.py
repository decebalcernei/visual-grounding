import torch
import clip
from baseline import yolo_inference, clip_inference
from util import calculate_IoU, compare_bbox, draw_bbox
from tqdm import tqdm
import numpy as np
from types import SimpleNamespace

device = "cuda" if torch.cuda.is_available() else "cpu"

def eval_loop_baseline(clip_model, clip_preprocess, yolo_model, data):
    correct = 0
    total = 0
    iou_array = []
    
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in tqdm(data, desc="Processing Test Data"):
            images = sample['image']
            for i, image in enumerate(images):
                cropped_images, bboxes = yolo_inference(yolo_model, image)
                pred_bbox = torch.tensor(clip_inference(clip_model, clip_preprocess, cropped_images, bboxes, sample['description'][i]), dtype=torch.float32)

                iou = calculate_IoU(pred_bbox.unsqueeze(0), sample['bbox'][i].unsqueeze(0))

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
    model.eval()
    all_ious = []

    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Evaluating"):
            images = sample["image"].to(device)
            descriptions = sample["description"].to(device)
            gt_bboxes = sample["bbox"].to(device)

            predicted_bboxes = model(images, descriptions)

            ious = calculate_IoU(gt_bboxes, predicted_bboxes)

            all_ious.extend(ious.tolist())

    mean_iou = sum(all_ious) / len(all_ious)
    return mean_iou
