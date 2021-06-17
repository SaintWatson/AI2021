##
""" Import package"""
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.HyperParameter import *
##
""" Loading dataset """
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.datasets import ImageFolder

def prepare_dataloader():

    first_data = ImageFolder('datasets/train', transform=TRAIN_TFM)
    second_data = ImageFolder('datasets/val', transform=TEST_TFM)
    third_data = ImageFolder('datasets/test', transform=TEST_TFM)

    # Because first_dataset is far larger than the second one, 
    # we split some data from first one to it.
    n_val = int(len(first_data) * VAL_RATIO)
    idx = np.random.permutation(len(first_data))

    train_data = Subset(first_data, idx[n_val:])
    val_data = ConcatDataset([second_data, Subset(first_data, idx[:n_val])])
    test_data = third_data

    print(f'train data: {len(train_data)}')
    print(f'valid data: {len(val_data)}')
    print(f'test data: {len(test_data)}')

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    del first_data, second_data, third_data, train_data, val_data, test_data

    return train_loader, val_loader, test_loader

##
""" Setup network """
from torch.optim import AdamW, lr_scheduler
from torchvision.models.resnet import resnet152

def prepare_network(load=False):

    if load:
        pass
    else:
        model = resnet152(pretrained=True).to(DEVICE)
        n_feats = model.fc.in_features
        model.fc = nn.Linear(n_feats, 2)

    print('Load the model')

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), 
                        lr=LEARNING_RATE,
                        weight_decay=1e-5)

    scheduler = lr_scheduler.StepLR(optimizer,
                                    step_size=7,
                                    gamma=0.1)

    return model, criterion, optimizer, scheduler

##
""" Setup training routine """
def train(model, criterion, optimizer, scheduler, dataloader, phase):

    if phase == "Train":
        model.train()
    elif phase == "Valid":
        model.eval()
    else:
        print("Not exist this mode")
        return

    running_loss = 0.0
    running_correct = 0
    model = model.to(DEVICE)

    for batch in tqdm(dataloader):

        imgs, labels = batch

        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        pred = model(imgs)
        loss = criterion(pred, labels)

        if phase == "Train":
            loss.backward()
            optimizer.step()
            scheduler.step()

        _, result = torch.max(pred, 1)
        running_correct += torch.sum(result == labels.data)
        running_loss += loss.item() * imgs.size(0)
    
    epoch_acc = running_correct / len(dataloader.dataset)
    epoch_loss = running_loss / len(dataloader.dataset)

    return epoch_loss, epoch_acc


    
##
""" Training part """
train_loader, val_loader, test_loader = prepare_dataloader()
model, criterion, optimizer, scheduler = prepare_network()


train_loss_record = []
train_acc_record = []
valid_loss_record = []
valid_acc_record = []

for epoch in range(1):

    loss, acc = train(model, criterion, optimizer, scheduler, train_loader, "Train")
    train_loss_record.append(loss)
    train_acc_record.append(acc)

    print(f'Epoch: {epoch+1}/{N_EPOCH} | loss = {loss:.5f}, acc = {acc:.5f}')
    torch.save(model, "checkpoint/model.pth")

    loss, acc = train(model, criterion, optimizer, scheduler, train_loader, "Valid")
    valid_loss_record.append(loss)
    valid_acc_record.append(acc)

