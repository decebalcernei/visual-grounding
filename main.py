from util import create_merged_df, VisualGroundingRefcocog, get_dataloader, modified_clip_preprocess, resize_bbox
from functions import eval_loop_baseline, train_loop, eval_loop
import warnings
from tqdm import tqdm
from model import VisualLanguisticTranformer
import clip
import torch
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    DEVICE = 'cuda'
    batch_size = 32
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

    #result, mean_iou = eval_loop(test_dataset)

    #print(f"Accuracy is {result}  -----  Mean IoU is {mean_iou}")

    clip_model, _ = clip.load("RN50", device=DEVICE)
    num_encoders = 6
    model = VisualLanguisticTranformer(num_encoders, clip_model)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters()) 
    n_epochs = 99
        

    for epoch in tqdm(range(1,n_epochs)):
        loss = train_loop(model, train_dataloader, optimizer, device=DEVICE)
        print(f'loss at epoch {epoch} is {np.asarray(loss).mean()}')
        if epoch % 3 == 0: # We check the performance every 3 epochs
            mean_iou = eval_loop(model, val_dataloader, device=DEVICE)
            print(f'mean_iou at epoch {epoch} = {mean_iou}')


if __name__ == "__main__":
    main()

    