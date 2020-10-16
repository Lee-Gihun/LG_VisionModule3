import torch
from materials.utils import *
import time
import warnings
warnings.filterwarnings("ignore")

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, criterion, optimizer, scheduler=None, 
          num_epochs=200, grad_clip=None, print_freq=1, 
          save_name='test', device=default_device):
    
    model.train()
    
    for epoch in range(num_epochs):
        batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
        start = time.time()
    
        for i, (images, boxes, labels, _) in enumerate(dataloader):
            data_time.update(time.time()-start)

            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # forward pass
            pred_locs, pred_scores = model(images)
            loss = criterion(pred_locs, pred_scores, boxes, labels)
            if loss > 100:
                continue
            # backward pass
            optimizer.zero_grad()
            loss.backward()

            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)

            optimizer.step()

            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(dataloader),
                                                                      batch_time=batch_time,
                                                                      data_time=data_time, loss=losses))
        if scheduler is not None:
            scheduler.step()
            
            
    torch.save(model.state_dict(), './{}_{}.pth'.format(save_name, round(losses.val, 3)))
