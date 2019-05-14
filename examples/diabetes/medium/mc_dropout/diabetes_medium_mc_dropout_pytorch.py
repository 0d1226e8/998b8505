
import numpy as np

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

MODEL_CONFIG=[16, 'M', 16, 16, 'M', 32, 32, 'M', 64, 64, 64, 64, 'M', 128, 128, 128, 128]

class MyModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv_layers = self.make_layers(MODEL_CONFIG)
        self.fc = nn.Linear(256, 1)
        
    def make_layers(self, config):
        layers = []
        in_channels = 3
        for idx, c in enumerate(config):
            if c == 'M':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
            else:
                stride = 2 if (idx == 1) else 1

                layers += [nn.Conv2d(in_channels, c, 3, stride, padding=1),
                           nn.LeakyReLU(inplace=True), 
                           nn.Dropout(0.1)]

                in_channels = c
                
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv_layers(x)
        
        mean_pool = F.adaptive_avg_pool2d(x, 1).squeeze()
        max_pool = F.adaptive_max_pool2d(x, 1).squeeze()
        x = torch.cat([mean_pool, max_pool], dim=1)
        
        x = self.fc(x)
        
        return x
		
def accuracy(output, target):
	acc = torch.sum((output >= 0) == target.type(torch.cuda.ByteTensor)).type(torch.cuda.FloatTensor) / output.shape[0]
	return acc
		
import bdlb
from bdlb.tasks import DiabetesMedium, DiabetesRealWorld

dtask_medium = DiabetesMedium('./data/diabetes')
# dtask_realworld = DiabetesRealWorld("./data/diabetes")

train_dataset, eval_dataset, test_dataset = dtask_medium.get_pytorch_datasets()

# We can now turn them into PyTorch dataloader as we normally would with any other datasets:
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(eval_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
optimiser = torch.optim.SGD(
    model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-5, nesterov=True
)

for epoch in range(150):
	model.train()
	start = time.time()
	for idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device).float()
		optimiser.zero_grad()
		output = model(data).squeeze()
		loss = F.binary_cross_entropy_with_logits(output, target)
		loss.backward()
		optimiser.step()
		
		if idx % 20 == 0:
			acc = accuracy(output, target)
			print("Batch {}/{} Loss: {} Acc: {}".format(idx, len(train_loader), loss.item(), acc.detach().cpu().numpy()))
	end = time.time()
	print('Time for one epoch: {:.1f} secs'.format(end-start))
	
	if epoch % 2 == 0:
		model.eval()
		with torch.no_grad():
			loss = 0
			acc = 0
			num_batches = len(valid_loader)
			for idx, (data, target) in enumerate(valid_loader):
				data, target = data.to(device), target.to(device).float()
				output = model(data).squeeze()
				loss += F.binary_cross_entropy_with_logits(output, target) / num_batches
				acc += accuracy(output, target) / num_batches
			print("EPOCH {}".format(epoch))
			print("=================================")
			print("Loss: {} Acc: {}".format(loss, acc))
