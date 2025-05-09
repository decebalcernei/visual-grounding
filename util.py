import pandas as pd
import json
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
import torch
import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as F


def create_merged_df(pickle_file_path, annotations_file_path):
    """
        instances.json has 208960 elements
        refs(umd).p    has 49822  elements
        we return the merged dataframe that has the same elements as the ref file (in the case of the umd one 49822)
    """
    annotations_file_path = 'dataset/refcocog/annotations/instances.json'
    pickle_file_path = 'dataset/refcocog/annotations/refs(umd).p'


    with open(annotations_file_path, 'r') as f:
        annotations = json.load(f)["annotations"]
        annotations = pd.DataFrame(annotations)

    # we remove also the image_id since they will be merged on the bbox id which is more specific
    columns_to_remove = ["segmentation", "area", "iscrowd", "category_id", "image_id"] # possible: segmentation, area, iscrowd, image_id, bbox, category_id, id
    annotations = annotations.drop(columns=columns_to_remove)


    partition = pd.read_pickle(pickle_file_path) # read Pickle file
    partition = pd.DataFrame.from_records(partition) # convert it to a dataframe

    
    columns_to_remove = ["category_id", "sent_ids", "ref_id", "image_id"] # possible: image_id, split, sentences, file_name, category_id, ann_id, sent_ids, ref_id
    partition = partition.drop(columns=columns_to_remove)


    # We merge on id == ann_id which are the descriptions of the bboxes
    merged = pd.merge(annotations, partition, left_on='id', right_on='ann_id', how='inner')

    # we decide, for now, to use the first sentence/description of the box. Some bboxes have several descriptions for the same bbox
    #print(merged.head()['sentences'])
    merged['sentences'] = merged['sentences'].apply(lambda x: x[0]['sent'])
    #print(merged.head()['sentences'])


    #now we refine the file name by adding the root dir and by cleaning the name (in particular it has an additional number before the extension which the images does not have)

    merged['file_name'] = merged['file_name'].apply(modify_filename)

    # we don't need the id anymore
    columns_to_remove = ["ann_id", "id"]
    merged = merged.drop(columns=columns_to_remove)

    # merged has ['bbox', 'split', 'sentences', 'filename'], everything we need and nothing more. Note that we have 3 splits, train, val and test.
    return merged


def modify_filename(file_name):
    """
        function for modifying the file_name of each image in the dataframe
    """

    # Remove the last number before the extension
    base_name = '_'.join(file_name.split('_')[:-1]) + '.jpg'
    
    # Add the root dictory of the dataset
    new_name = f'dataset/refcocog/images/{base_name}'
    
    return new_name


class VisualGroundingRefcocog(data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, tokenizer, modified_clip_preprocess=None, bbox_transform=None):
        
        self.images = dataset['file_name'].tolist()
        self.descriptions = dataset['sentences'].tolist()
        self.bboxes = [xywh2xyxy(bbox) for bbox in dataset['bbox'].tolist()]
        self.transform = modified_clip_preprocess
        self.bbox_transform = bbox_transform
        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):

        image = Image.open(self.images[idx])
        description = self.descriptions[idx]
        bbox = self.bboxes[idx]

        # We directly apply the CLIP's modified preprocess to the images
        """
            Why we modify the CLIP's preprocess? Because it performs a center crop of the image
            and we dont want that. We want to keep the whole image and not risk to cut away our
            object of interest.
        """
        #path_saving_folder = "/home/dec/uni/dl/visual-grounding/tests"
        #original_path = path_saving_folder + "/original_no_normalization.png"
        #resized_path = path_saving_folder + "/resized_no_normalization.png"

        description = self.tokenizer(description).squeeze() # (1, 77) -> (77)

        if self.transform:
            original_width, original_height = image.size
            image = self.transform(image)

        if self.bbox_transform:
            bbox = self.bbox_transform(bbox, (original_width, original_height), (224, 224))


        sample = {
            'image': image,
            'description': description,
            'bbox': bbox,
        }

        return sample


def get_dataloader(dataset, batch_size):

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        #num_workers=2
    )

    return data_loader


def calculate_IoU(x_bbox, y_bbox):
    """
    Calculate IoU between two batches of bounding boxes.
    x_bbox, y_bbox: tensors of shape (batch_size, 4)
    Returns: tensor of IoU values, shape (batch_size)
    """

    # get intersection's coordinates
    inter_xmin = torch.max(x_bbox[:, 0], y_bbox[:, 0])
    inter_ymin = torch.max(x_bbox[:, 1], y_bbox[:, 1])
    inter_xmax = torch.min(x_bbox[:, 2], y_bbox[:, 2])
    inter_ymax = torch.min(x_bbox[:, 3], y_bbox[:, 3])

    # clamp to zero when no intersection
    inter_w = (inter_xmax - inter_xmin).clamp(min=0)
    inter_h = (inter_ymax - inter_ymin).clamp(min=0)
    intersection = inter_w * inter_h

    # areas
    area_x = (x_bbox[:, 2] - x_bbox[:, 0]) * (x_bbox[:, 3] - x_bbox[:, 1])
    area_y = (y_bbox[:, 2] - y_bbox[:, 0]) * (y_bbox[:, 3] - y_bbox[:, 1])
    union = area_x + area_y - intersection

    iou = intersection / union.clamp(min=1e-6)
    return iou


def xywh2xyxy(bbox):
    '''
    refcocog labels are in the format x y w h
    yolo are xmin ymin xmax ymax

    Input format:
            bbox         = [x, y, w, h]
        Output:
            updated_bbox = [x, y, x, y]
    '''
    xmin, ymin, w, h = bbox
    
    xmax = xmin + w
    ymax = ymin + h
    
    updated_bbox = [xmin, ymin, xmax, ymax]

    return updated_bbox


def draw_bbox(image, bbox, color, save_path=None, caption=None):
    """
    image is a tensor (C, H, W) and bbox is an array of the 4 coordinates
    """
    xmin, ymin, xmax, ymax = bbox

    if isinstance(image, torch.Tensor):
        image = F.to_pil_image(image)

    draw = ImageDraw.Draw(image)
    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
    if caption is not None:
        draw.text((5, 5), caption, fill="white")
    
    if save_path is not None:
        image.save(save_path)
    else:
        image.show()



def compare_bbox(image, pred_bbox, label_bbox, caption=None, color1="green", color2="red"):
    #img = Image.open(image_path)
    
    xmin1, ymin1, xmax1, ymax1 = label_bbox
    xmin2, ymin2, xmax2, ymax2 = pred_bbox
    
    draw = ImageDraw.Draw(image)
    draw.rectangle([xmin1, ymin1, xmax1, ymax1], outline=color1, width=3)
    draw.rectangle([xmin2, ymin2, xmax2, ymax2], outline=color2, width=3)
    if caption is not None:
        draw.text((5, 5), caption, fill="white")
    
    image.show()


def modified_clip_preprocess():
    """
    The original CLIP's preprocess is the following:
    Compose(
        Resize(size=224, interpolation=bicubic, max_size=None, antialias=True)
        CenterCrop(size=(224, 224))
        <function _convert_image_to_rgb at 0x744263e27ba0>
        ToTensor()
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    )
    """
    """
    Note that torchvision.transforms.Compose expects callables (i.e., functions or objects you can "call") as transformations
    
    transforms.Lambda(lambda img: img.convert("RGB")) is equivalent to
    
    def convert_rgb(img):
        return img.convert("RGB")

    transforms.Lambda(convert_rgb)
    """
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224), interpolation=Image.BICUBIC, antialias=True),
        transforms.Lambda(lambda img: img.convert("RGB")),  # to rgb
        transforms.ToTensor(), # from PIL image to tensor
        #transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                  #std=(0.26862954, 0.26130258, 0.27577711))

    ])

    return transform


def resize_bbox(bbox, original_size, new_size):
    original_width, original_height = original_size
    new_width, new_height = new_size

    scale_x = new_width / original_width
    scale_y = new_height / original_height

    x1, y1, x2, y2 = bbox
    
    resized_bbox = [
        x1 * scale_x,
        y1 * scale_y,
        x2 * scale_x,
        y2 * scale_y
    ]
    # To tensor
    resized_bbox = torch.tensor(resized_bbox, dtype=torch.float32)

    return resized_bbox