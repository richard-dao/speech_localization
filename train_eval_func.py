import torch
import torch.nn as nn
from tqdm import tqdm
from mae import calc_angular_errors
from config import config
from torch.nn.parallel import DistributedDataParallel as DDP
from snntorch import utils as snn_utils

def train_func(model, dataloader, optimizer, loss_func, device, rank):
    model.train()
    total_loss = 0
    all_errors = []

    i = 0
    for features, labels in tqdm(dataloader, desc='Training', leave=True, disable=rank != 0):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if config["model_config"]["use_snn"]:
            if isinstance(model, DDP) or isinstance(model, nn.DataParallel):
                snn_utils.reset(model.module.features)
            else:
                snn_utils.reset(model.features)
        
        outputs = model(features)
        if i % 10 == 0:
            print(outputs, labels)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(loss)

        total_loss += loss.item() * labels.size(0)
        errors = calc_angular_errors(
            labels[:, 0], labels[:, 1], outputs[:, 0], outputs[:, 1]
        )
        all_errors.extend(errors)

    if rank == 0:
        average_loss = total_loss / len(dataloader.dataset)
        all_errors_tensor = torch.stack(all_errors)
        overall_min_error = torch.min(all_errors_tensor)
        overall_median_error = torch.median(all_errors_tensor)
        overall_max_error = torch.max(all_errors_tensor)
        return average_loss, overall_min_error.item(), overall_median_error.item(), overall_max_error.item()
    else:
        return None  
    
def evaluate_func(model, dataloader, loss_func, device, rank):
    model.eval()
    total_loss = 0
    all_errors = []
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc='Evaluation', leave=True, disable=rank != 0):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = loss_func(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            errors = calc_angular_errors(
                labels[:, 0], labels[:, 1], outputs[:, 0], outputs[:, 1]
            )
            all_errors.extend(errors)

    if rank == 0:
        average_loss = total_loss / len(dataloader.dataset)
        all_errors_tensor = torch.stack(all_errors)
        overall_min_error = torch.min(all_errors_tensor)
        overall_median_error = torch.median(all_errors_tensor)
        overall_max_error = torch.max(all_errors_tensor)
        return average_loss, overall_min_error.item(), overall_median_error.item(), overall_max_error.item()
    else:
        return None
