import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

from materials.DetectionNet import DetectionNet, create_prior_boxes
from materials.MultiboxLoss import MultiBoxLoss
from materials.datasets import PascalVOCDataset
from materials.utils import *
from materials.train import train

import time
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = PascalVOCDataset(data_folder='./data/VOC', split='TRAIN')
dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, 
                              collate_fn=dataset.collate_fn, num_workers=4)

model = DetectionNet(n_classes=20, unfreeze_keys='all', use_bias=True).to(device)
criterion = MultiBoxLoss(priors_cxcy=create_prior_boxes(), threshold=0.5, neg_pos_ratio=0.3, alpha=1.0)

num_epochs = 40
lr, momentum, weight_decay = 1e-3, 0.9, 5e-4

biases, not_biases = [], []

for param_name, param in model.named_parameters():
    if param.requires_grad:
        if param_name.endswith('.bias'):
            biases.append(param)
        else:
            not_biases.append(param)
optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                            lr=lr, momentum=momentum, weight_decay=weight_decay)

#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=False)
scheduler = scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00005)

    
def main():
    train(model, dataloader, criterion, optimizer, scheduler, num_epochs, grad_clip=None, print_freq=500, save_name='our_model')
    

if __name__ == '__main__':
    main()