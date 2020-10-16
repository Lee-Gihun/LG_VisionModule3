import torch
from torchvision import transforms
from materials.utils import *
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model에 입력하기 위한 기본 Transformation을 설정합니다.
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(original_image, model, min_score=0.5, max_overlap=0.3, top_k=5, suppress=None):
    # model을 test mode로 설정합니다.
    model.eval()
    
    # Transform을 수행합니다.
    image = normalize(to_tensor(resize(original_image)))

    # 지정된 device에서 연산합니다.
    image = image.to(device)

    # model에 대한 forward pass를 연산합니다.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))
    prior_boxes = create_prior_boxes()
    priors_cxcy = xy_to_cxcy(prior_boxes)
    
    # detection 결과를 연산합니다.
    det_boxes, det_labels, det_scores = detect_objects(predicted_locs, predicted_scores, priors_cxcy, min_score=min_score, max_overlap=max_overlap, top_k=top_k, n_classes=20)

    # 결과값을 cpu에 저장합니다.
    det_boxes = det_boxes[0].to('cpu')

    # 원래의 image 크기로 변환합니다.
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # 라벨 값(0~20)에 대응하는 라벨 이름을 가져옵니다.
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    
    # background 뿐이면 이미지를 그대로 출력합니다.
    if det_labels == ['background']:
        return original_image

    # 이미지 위에 bounding box와 라벨을 표시합니다.
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()
    
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])
        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    return annotated_image




def ssd_detect(original_image, model, min_score, max_overlap, top_k, suppress=None, weapon=True):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    
    if weapon:
        det_labels = ['background', 'weapon']

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        if not weapon:
            draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
            draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
                det_labels[i]]) 
        else:
            draw.rectangle(xy=box_location, outline='black')
            draw.rectangle(xy=[l + 1. for l in box_location], outline='black')  

        # Text

        if not weapon:
            text_size = font.getsize(det_labels[i].upper())
            text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
            textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                                box_location[1]]
            draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
            draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                      font=font)
        else:
            text_size = font.getsize('weapon'.upper())
            text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
            textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                                box_location[1]]
            draw.rectangle(xy=textbox_location, fill='black')
            draw.text(xy=text_location, text='weapon'.upper(), fill='white',
                      font=font)
    del draw

    return annotated_image
