import unicodedata
import re
import torch
import numpy as np
import time
import math
import random
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from random import sample
import torch.nn as nn
import torch.optim as optim
import datetime

class LinearModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.1):
        super(LinearModel, self).__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.input(x)
        x = self.hidden(x)
        x = self.output(x)
        return x
    
def loss_fn(outputs, target, criterion):
    # Calculate reconstruction loss
    recon_loss = criterion(
        outputs.view(-1, outputs.size(-1)),
        target.view(-1)
    )
    
    # Get the predicted labels by taking the argmax of the outputs
    _, predicted = torch.max(outputs, dim=-1)
    
    # Flatten the predictions and targets
    predicted = predicted.view(-1)
    target = target.view(-1)
    
    # Calculate the number of correct predictions
    correct = (predicted == target).sum().item()
    
    # Calculate the number of wrong predictions
    wrong = (predicted != target).sum().item()
    
    return recon_loss, correct, wrong

def train_epoch(loader, model, criterion, optimizer, scheduler, epoch):
    num_correct = 0
    num_wrong = 0
    total_loss = 0
    for data in loader:
        input_tensor, target_tensor = data
        outputs = model(input_tensor)
        loss,right, wrong = loss_fn(outputs, target_tensor, criterion)
        loss.backward()
        optimizer.step()
        num_correct += right
        num_wrong = wrong
        total_loss += loss.item()
    
    print("Train Epoch " + str(epoch) + " accuracy: " + str(num_correct/(num_correct + num_wrong)))
    print("Train Epoch " + str(epoch) + " loss: " + str(total_loss))

    
    scheduler.step()


def test_epoch(loader, model, epoch):
    num_correct = 0
    num_wrong = 0
    total_loss = 0
    for data in loader:
        input_tensor, target_tensor = data
        outputs = model(input_tensor)
        loss,right, wrong = loss_fn(outputs, target_tensor, criterion)
        num_correct += right
        num_wrong = wrong
        total_loss += loss.item()

    
    print("Test Epoch " + str(epoch) + " accuracy: " + str(num_correct/(num_correct + num_wrong)))
    print("Test Epoch " + str(epoch) + " loss: " + str(total_loss))


start = datetime.datetime.now()
file = open("data_blca.csv", mode='r')
info = []
for line in file:
    line = line.strip()
    parts = line.split(',')
    designation = int(parts[1])

    info_part = (np.array(parts[2:]).astype(np.float64), designation)

    info.append(info_part)

random.shuffle(info)

labels = [label[1] for label in info]
data = [label[0] for label in info]
num_data = len(data[0])
num_epochs = 201

train_labels = torch.tensor(labels[0:4*int(len(labels) / 5)], dtype=torch.long)
train_data = torch.tensor(data[0:4*int(len(labels) / 5)], dtype=torch.float)



test_labels = torch.tensor(labels[4*int(len(labels) / 5):], dtype=torch.long)
test_data = torch.tensor(data[4*int(len(labels) / 5):], dtype=torch.float)
train_data = TensorDataset(train_data, train_labels)
test_data = TensorDataset(test_data, test_labels)
criterion = nn.NLLLoss()

train_dataloader = DataLoader(train_data, batch_size=80, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=80, shuffle=True)

model = LinearModel(num_data, 500, 2)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print("Number of Parameters: " + str(params))
for i in range(num_epochs):
    train_epoch(train_dataloader, model, criterion, optimizer, scheduler, i)

    if(i % 5 == 0):
        test_epoch(test_dataloader, model, i)

end = datetime.datetime.now()

print("Start: " + str(start))
print("End: " + str(end))
# define model.

# define training.