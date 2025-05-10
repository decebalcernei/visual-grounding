from util import create_merged_df, VisualGroundingRefcocog, get_dataloader, modified_clip_preprocess, resize_bbox, init_weights
from functions import train_loop, eval_loop, eval_loop_baseline
import warnings
from tqdm import tqdm
from model import VisualLanguisticTranformer
import clip
import torch
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    DEVICE = 'cuda'
    batch_size = 8
    annotations_file_path = 'dataset/refcocog/annotations/instances.json'
    pickle_file_path = 'dataset/refcocog/annotations/refs(umd).p'
    whole_df = create_merged_df(pickle_file_path, annotations_file_path)
    
    # split the whole dataframe in train, val, test
    train_df = whole_df.loc[whole_df['split'] == 'train']
    val_df   = whole_df.loc[whole_df['split'] == 'val']
    test_df  = whole_df.loc[whole_df['split'] == 'test']

    image_transform = modified_clip_preprocess()
    bbox_transform = resize_bbox
    tokenizer = clip.tokenize
    train_dataset = VisualGroundingRefcocog(train_df, tokenizer, image_transform, bbox_transform)
    val_dataset = VisualGroundingRefcocog(val_df, tokenizer, image_transform, bbox_transform)
    test_dataset = VisualGroundingRefcocog(test_df, tokenizer, image_transform, bbox_transform)# has 5024 elements
    
    train_dataloader = get_dataloader(train_dataset, batch_size)
    val_dataloader = get_dataloader(val_dataset, batch_size)
    test_dataloader = get_dataloader(test_dataset, batch_size)

    clip_model, _ = clip.load("RN50", device=DEVICE)
    num_encoders = 6
    model = VisualLanguisticTranformer(num_encoders, clip_model)
    # we apply the init_weights function to initialize the projection layers -> speed up training
    # we start with better weights.
    model.apply(init_weights)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters()) 
    n_epochs = 21
        

    for epoch in tqdm(range(1,n_epochs)):
        loss = train_loop(model, train_dataloader, optimizer, device=DEVICE)
        print(f'loss at epoch {epoch} is {np.asarray(loss).mean()}')
        if epoch % 3 == 0: # We check the performance every 3 epochs
            mean_iou = eval_loop(model, val_dataloader, device=DEVICE)
            print(f'mean_iou at epoch {epoch} = {mean_iou}')
    mean_iou = eval_loop(model, test_dataloader, device=DEVICE)
    print(f'mean iou on test set is {mean_iou}')



def evaluate_baseline(data, device, modified_preprocess=None):
    clip_model, clip_preprocess = clip.load("RN50", device=device)
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, _verbose=False)
    if modified_preprocess is None:
        modified_preprocess = clip_preprocess
    
    accuracy, mean_iou = eval_loop_baseline(clip_model, modified_preprocess, yolo_model, data)

    return accuracy, mean_iou


if __name__ == "__main__":
    main()

    