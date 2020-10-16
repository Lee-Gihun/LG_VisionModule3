import torch
import torch.nn as nn
from materials.models.EfficientNet import EfficientNet

class BaseConvolution(nn.Module):
    def __init__(self, unfreeze_keys=['15', 'head', 'bn1']):
        """
        Pre-trained EfficientNet을 불러와 BaseConv를 구성합니다,
        """        
        super(BaseConvolution, self).__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b0')
        
        # 모델의 모든 parameter를 학습이 되지 않도록 합니다.
        for params in self.base.parameters():
            params.requires_grad = False
        
        # `unfreeze_keys`에 해당하는 parameter는 학습이 가능하도록 합니다.
        for name, params in self.base.named_parameters():
            for key in unfreeze_keys:
                if key in str(name):
                    params.requires_grad = True
        
        
    def forward(self, x):
        """
        BaseConvolution을 통과하여 나온 3개의 featuremap을 리턴합니다. 
        (tensor) x: (N, C, H, W)의 Image Data.
        """
        # base conv 연산을 수행합니다.
        mid, end = self.base(x)
        
        return mid, end
    
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxConvolution(nn.Module):
    def __init__(self, use_bias=False):
        super(AuxConvolution, self).__init__()
        self.conv_a1 = nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0, bias=use_bias)
        self.conv_a2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=use_bias)
        
        self.conv_b1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=use_bias)
        self.conv_b2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias)
        
        self.conv_c1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=use_bias)
        self.conv_c2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, bias=use_bias)
        
        self.conv_d1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=use_bias)
        self.conv_d2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, bias=use_bias)
        
        self.init__conv2d(use_bias)
        
        
    def init__conv2d(self, use_bias=False):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                if use_bias:
                    nn.init.constant_(c.bias, 0.)
        
        
    def forward(self, end_features):
        out = F.relu(self.conv_a1(end_features))
        out = F.relu(self.conv_a2(out))
        features_a = out
        
        out = F.relu(self.conv_b1(out))
        out = F.relu(self.conv_b2(out))
        features_b = out
        
        out = F.relu(self.conv_c1(out))
        out = F.relu(self.conv_c2(out))
        features_c = out
        
        out = F.relu(self.conv_d1(out))
        out = F.relu(self.conv_d2(out))
        features_d = out
        
        return features_a, features_b, features_c, features_d
    
    
import torch.nn as nn
import torch.nn.functional as F

class PredictionConvolution(nn.Module):
    def __init__(self, n_classes=20, use_bias=False):
        super(PredictionConvolution, self).__init__()
        self.n_classes = n_classes

        n_boxes = {'mid': 6, 'end': 6, 'a': 6,
                   'b': 6, 'c': 4, 'd': 4}

        self.loc_mid = nn.Conv2d(112, n_boxes['mid'] * 4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.loc_end = nn.Conv2d(1280, n_boxes['end'] * 4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.loc_a = nn.Conv2d(512, n_boxes['a'] * 4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.loc_b = nn.Conv2d(256, n_boxes['b'] * 4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.loc_c = nn.Conv2d(256, n_boxes['c'] * 4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.loc_d = nn.Conv2d(256, n_boxes['c'] * 4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        
        self.cls_mid = nn.Conv2d(112, n_boxes['mid'] * n_classes, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.cls_end = nn.Conv2d(1280, n_boxes['end'] * n_classes, kernel_size=3, stride=1, padding=1, bias=use_bias)  
        self.cls_a = nn.Conv2d(512, n_boxes['a'] * n_classes, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.cls_b = nn.Conv2d(256, n_boxes['b'] * n_classes, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.cls_c = nn.Conv2d(256, n_boxes['c'] * n_classes, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.cls_d = nn.Conv2d(256, n_boxes['d'] * n_classes, kernel_size=3, stride=1, padding=1, bias=use_bias)
        
        self.init__conv2d(use_bias)
        
    def init__conv2d(self, use_bias=False):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                if use_bias:
                    nn.init.constant_(c.bias, 0.)
                    
                    
    def forward(self, mid, end, a, b, c, d):
        
        # predict bounding boxes
        l_mid = self.loc_mid(mid)
        l_end = self.loc_end(end)
        l_a = self.loc_a(a)
        l_b = self.loc_b(b)
        l_c = self.loc_c(c)
        l_d = self.loc_d(d)

        l_mid = self._box_align(l_mid, mode='loc')
        l_end = self._box_align(l_end, mode='loc')
        l_a = self._box_align(l_a, mode='loc')
        l_b = self._box_align(l_b, mode='loc')
        l_c = self._box_align(l_c, mode='loc')
        l_d = self._box_align(l_d, mode='loc')

        # predict classes of boxes
        c_mid = self.cls_mid(mid)
        c_end = self.cls_end(end)
        c_a = self.cls_a(a)
        c_b = self.cls_b(b)
        c_c = self.cls_c(c)
        c_d = self.cls_d(d)

        c_mid = self._box_align(c_mid, mode='cls')
        c_end = self._box_align(c_end, mode='cls')
        c_a = self._box_align(c_a, mode='cls')
        c_b = self._box_align(c_b, mode='cls')
        c_c = self._box_align(c_c, mode='cls')
        c_d = self._box_align(c_d, mode='cls')
        
        locs = torch.cat([l_mid, l_end, l_a, l_b, l_c, l_d], dim=1)
        cls_scores = torch.cat([c_mid, c_end, c_a, c_b, c_c, c_d], dim=1)
        
        return locs, cls_scores
    
    def _box_align(self, tensor, mode='loc'):
        batch_size = tensor.size(0)
        tensor = tensor.permute(0, 2, 3, 1).contiguous()
        
        if mode == 'loc':
            tensor = tensor.view(batch_size, -1, 4)
        
        elif mode == 'cls':
            tensor = tensor.view(batch_size, -1, self.n_classes)
            
        return tensor

        