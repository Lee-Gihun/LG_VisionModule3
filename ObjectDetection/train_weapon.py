import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

from materials.DetectionNet import DetectionNet, create_prior_boxes
from materials.MultiboxLoss import MultiBoxLoss
from materials.datasets import WeaponDataset
from materials.utils import *
from materials.train import train
    
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

dataset = WeaponDataset(data_folder='./data/Weapon')
dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, 
                                  collate_fn=dataset.collate_fn, num_workers=4)

model = DetectionNet(n_classes=1, unfreeze_keys=['15','head','bn1'], use_bias=True).to(device)
criterion = MultiBoxLoss(priors_cxcy=create_prior_boxes(device), threshold=0.5, 
                             neg_pos_ratio=0.3, alpha=1.0, device=device)

num_epochs = 200
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
    from materials.train import train
    import warnings
    warnings.filterwarnings("ignore")

    train(model, dataloader, criterion, optimizer, scheduler, num_epochs, 
          grad_clip=None, print_freq=50, save_name='weapon', device='cuda:1')    

if __name__ == '__main__':
    main()