import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import time

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
 
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.act2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
 
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.act3 = nn.Tanh()
 
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(1*1*120, 84)
        self.act4 = nn.Tanh()
        self.fc2 = nn.Linear(84, 10)
        
    def forward(self, x):
        # input 1x28x28, output 6x28x28
        x = self.act1(self.conv1(x))
        # input 6x28x28, output 6x14x14
        x = self.pool1(x)
        # input 6x14x14, output 16x10x10
        x = self.act2(self.conv2(x))
        # input 16x10x10, output 16x5x5
        x = self.pool2(x)
        # input 16x5x5, output 120x1x1
        x = self.act3(self.conv3(x))
        # input 120x1x1, output 84
        x = self.act4(self.fc1(self.flat(x)))
        # input 84, output 10
        x = self.fc2(x)
        return x

mps_device = torch.device("mps:0")

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,), (128,)),
])
train = torchvision.datasets.MNIST('data',train=True,download=True,transform=transform)
test = torchvision.datasets.MNIST('data',train=False,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=100)
testloader = torch.utils.data.DataLoader(test, shuffle=True, batch_size=100)


model = LeNet5()
model = model.to(mps_device)

optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()
 
n_epochs = 10
# time the epoch
for epoch in range(n_epochs):
    # time the epoch
    start = time.time()
    model.train()
    for i, (X_batch, y_batch) in enumerate(trainloader):
        X_batch, y_batch = X_batch.to(mps_device), y_batch.to(mps_device)
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    acc = 0
    count = 0
    for i, (X_batch, y_batch) in enumerate(testloader):
        X_batch, y_batch = X_batch.to(mps_device), y_batch.to(mps_device)
        y_pred = model(X_batch)
        acc += (torch.argmax(y_pred, 1) == y_batch).float().sum()
        count += len(y_batch)
    acc = acc / count
    end = time.time()
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100), 
          "Time %.2f" % (end-start), "sec")