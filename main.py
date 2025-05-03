from PIL import Image, ImageDraw
from util import create_merged_df, VisualGroundingRefcocog, get_dataloader, create_transform
from functions import eval_loop, model_test
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    batch_size = 1
    annotations_file_path = 'dataset/refcocog/annotations/instances.json'
    pickle_file_path = 'dataset/refcocog/annotations/refs(umd).p'
    whole_df = create_merged_df(pickle_file_path, annotations_file_path)
    # split the whole dataframe in train, val, test
    
    #train_df = whole_df.loc[whole_df['split'] == 'train']
    #val_df   = whole_df.loc[whole_df['split'] == 'val']
    test_df  = whole_df.loc[whole_df['split'] == 'test']

    transform = create_transform()
    
    #train_dataset = VisualGroundingRefcocog(train_df)
    #val_dataset = VisualGroundingRefcocog(val_df)
    test_dataset = VisualGroundingRefcocog(test_df)#, transform=transform) # has 5024 elements
    
    #train_dataloader = get_dataloader(train_dataset, batch_size)
    #val_dataloader = get_dataloader(val_dataset, batch_size)
    test_dataloader = get_dataloader(test_dataset, batch_size)

    result, mean_iou = eval_loop(test_dataset)

    print(f"Accuracy is {result}  -----  Mean IoU is {mean_iou}")

    model_test(test_dataset)


if __name__ == "__main__":
    main()

    