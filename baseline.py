import torch
import clip
from PIL import Image, ImageDraw
from util import draw_bbox


device = "cuda" if torch.cuda.is_available() else "cpu"


def yolo_model(model, image):
    # Model
    # we will pass it to not load it each time
    #model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Inference
    results = model(image)

    # Results
    
    """
    results: xmin    ymin    xmax   ymax  confidence  class    name
    """
    bboxes = results.xyxy[0].cpu().numpy()
    
    bboxes = [bbox[:4] for bbox in bboxes] # take only the coordinates of the bbox

    
    cropped_images = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox

        # get the content of each bbox
        cropped_img = image.crop((xmin, ymin, xmax, ymax))
        cropped_images.append(cropped_img)

    return cropped_images, bboxes


def clip_model(model, preprocess, regions, bboxes, description):
    # Load the model and the preprocess
    # to not load it everytime we pass it
    #model, preprocess = clip.load("RN50", device=device)

    #print(preprocess)
    """
    This is CLIP's preprocess:
    Compose(
        Resize(size=224, interpolation=bicubic, max_size=None, antialias=True)
        CenterCrop(size=(224, 224))
        <function _convert_image_to_rgb at 0x744263e27ba0>
        ToTensor()
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    )
    """
    if regions == []: # in case yolo did not predict anything
        dummy_bbox = [0, 0, 0, 0]
        return dummy_bbox
    # Preprocess the images
    images = [preprocess(region).unsqueeze(0).to(device) for region in regions]

    image_batch = torch.cat(images, dim=0)
    description = "a photo of " + description
    description = clip.tokenize(description).to(device) # tokenize the description and move on gpu

    with torch.no_grad():
        image_features = model.encode_image(image_batch)
        text_features = model.encode_text(description)

        similarity = image_features @ text_features.T
        probs = similarity.softmax(dim=0).cpu()
        max_prob_index = torch.argmax(probs).numpy()
    
    pred_bbox = bboxes[max_prob_index]

    return pred_bbox
