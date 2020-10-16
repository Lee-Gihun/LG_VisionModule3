import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from .det_modules import BaseConvolution, AuxConvolution, PredictionConvolution
from .utils import *

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DetectionNet(nn.Module):
    def __init__(self, n_classes=20, unfreeze_keys=['15', 'head', 'bn1'], 
                 use_bias=False, device=default_device):
        super(DetectionNet, self).__init__()
        self.device = device
        self.n_classes = n_classes
        
        self.base = BaseConvolution(unfreeze_keys=unfreeze_keys)
        self.aux = AuxConvolution(use_bias=use_bias)
        self.pred = PredictionConvolution(n_classes=n_classes, use_bias=use_bias)            

        self.priors_cxcy = create_prior_boxes()
    
    
    def forward(self, images):
        mid, end = self.base(images)

        a, b, c, d = self.aux(end)
        
        locs, cls_scores = self.pred(mid, end, a, b, c, d)
        
        return locs, cls_scores
    
    
    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.
        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.
        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(self.device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress, (overlap[box] > max_overlap).byte())
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(self.device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(self.device))
                image_labels.append(torch.LongTensor([0]).to(self.device))
                image_scores.append(torch.FloatTensor([0.]).to(self.device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size
    
    
import torch
from math import sqrt

def create_prior_boxes(device=default_device):
    feature_dims = {'mid': 18, 'end': 9, 'a': 9,
                 'b': 5, 'c': 3, 'd' : 1}

    obj_scales = {'mid': 0.2, 'end': 0.2, 'a': 0.375,
                 'b': 0.55, 'c': 0.725, 'd' : 0.9}

    aspect_ratio = {'mid': [1., 2., 3., 0.5, 0.333],
                    'end': [1., 2., 3., 0.5, 0.333], 
                    'a': [1., 2., 3., 0.5, 0.333],
                    'b': [1., 2., 3., 0.5, 0.333], 
                    'c': [1., 2., 0.5], 
                    'd' : [1., 2., 0.5]}

    fmaps = list(feature_dims.keys())

    prior_boxes = []

    for k, fmap in enumerate(fmaps):
        for i in range(feature_dims[fmap]):
            for j in range(feature_dims[fmap]):
                cx = (j + 0.5) / feature_dims[fmap]
                cy = (i + 0.5) / feature_dims[fmap]

                for ratio in aspect_ratio[fmap]:
                    prior_boxes.append([cx, cy, 
                                        obj_scales[fmap] * sqrt(ratio),
                                        obj_scales[fmap] / sqrt(ratio)])

                if fmap in ['mid', 'end', 'a', 'b', 'c']:
                    additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k+1]])
                else:
                    additional_scale = 1.

                prior_boxes.append([cx, cy,
                                    additional_scale,
                                    additional_scale])

    prior_boxes = torch.FloatTensor(prior_boxes).to(device)
    prior_boxes.clamp_(0, 1)

    return prior_boxes