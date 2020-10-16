import torch
import os, glob, json, csv
from materials.utils import *
from PIL import Image
import numpy as np

data_path = os.path.abspath('./data/Weapon/')
anno_path = glob.glob(os.path.join(data_path, 'TrainData/*.txt'))

train_images, train_objects = [], []
n_objects = 0

for path in anno_path:

    with open(path, newline='') as f:
        boxes = []

        reader = csv.reader(f, delimiter=' ')
        for box in reader:
            box = [float(elem) for elem in box]
            boxes.append(box[1:])

    if len(boxes) == 0:
        continue
    
    img_path = path[:-3]+'jpg'
    # corrupted image
    if 'b043a132fb84b656' in img_path:
        continue
        
    train_images.append(img_path)
    image = Image.open(img_path, mode='r')
    w, h = image.size

    image_dim = torch.Tensor([w, h, w, h]).unsqueeze(0)    

    boxes = torch.Tensor(boxes)
    boxes = cxcy_to_xy(boxes)
    boxes = image_dim * boxes
    boxes = boxes.clamp(0, max(w,h))
    boxes = boxes.long()
    boxes = boxes.tolist()

    n_objects += len(boxes)
    
    anno_dict = {
        'boxes' : boxes,
        'labels' : [1 for x in range(len(boxes))],
        'difficulties': [0 for x in range(len(boxes))]}
    
    train_objects.append(anno_dict)
    
assert len(train_objects) == len(train_images)

with open('./data/Weapon/TRAIN_images.json', 'w') as j:
    json.dump(train_images, j)
    
with open('./data/Weapon/TRAIN_objects.json', 'w') as j:
    json.dump(train_objects, j)
    
with open('./data/Weapon/label_map.json', 'w') as j:
    label_map = {'background': 0, 'weapon': 1}
    json.dump(label_map, j)