import pandas as pd
import json
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
import torch
import torch.utils.data as data


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
    def __init__(self, dataset):
        
        self.images = dataset['file_name'].tolist()
        self.descriptions = dataset['sentences'].tolist()
        self.bboxes = []
        
        for bbox in dataset['bbox'].tolist():
            updated_bbox = xywh2xyxy(bbox)
            self.bboxes.append(updated_bbox)


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        
        image = image = Image.open(self.images[idx])
        description = self.descriptions[idx]
        bbox = self.bboxes[idx]

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
        num_workers=2
    )

    return data_loader


def calculate_IoU(x_bbox, y_bbox):

    xmin1, ymin1, xmax1, ymax1 = x_bbox
    xmin2, ymin2, xmax2, ymax2 = y_bbox

    # get intersection's coordinates
    inter_xmin = max(xmin1, xmin2)
    inter_ymin = max(ymin1, ymin2)
    inter_xmax = min(xmax1, xmax2)
    inter_ymax = min(ymax1, ymax2)

    # there is no intersection
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        iou = 0

    else:
        # calculate the intersection area
        intersection_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)

        # calculate the area of each bbox
        area_bbox1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area_bbox2 = (xmax2 - xmin2) * (ymax2 - ymin2)

        # calculate the union area
        union_area = area_bbox1 + area_bbox2 - intersection_area

        # calculate the intersectio over union
        iou = intersection_area / union_area
    
    return iou


def xywh2xyxy(bbox):
    '''
    refcocog labels are in the format x y w h
    yolo are xmin ymin xmax ymax
    '''
    xmin, ymin, w, h = bbox
    
    xmax = xmin + w
    ymax = ymin + h
    
    updated_bbox = [xmin, ymin, xmax, ymax]

    return updated_bbox


def draw_bbox(image, bbox, color, caption=None):
    #img = Image.open(image_path)
    
    xmin, ymin, xmax, ymax = bbox
    
    draw = ImageDraw.Draw(image)
    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
    if caption is not None:
        draw.text((5, 5), caption, fill="white")
    
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
