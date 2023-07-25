import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import matplotlib.pyplot as plt

from model import Model
from arguments import get_arguments
from utils import transformations

# detect if MPS is available
mps_device = torch.device("mps")

#get arguments from arguments.py
args = get_arguments()
print(args)

# normalization used in MNIST dataset
transform = transformations()

#seed the random number generator
torch.manual_seed(args.seed)
train = torchvision.datasets.MNIST('data1',train=True,download=True,transform=transform)
test = torchvision.datasets.MNIST('data1',train=False,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(train, shuffle=True,
                                          batch_size=args.batch_size)
testloader = torch.utils.data.DataLoader(test, shuffle=True,
                                         batch_size=args.batch_size)
print("Image size: ", train[0][0].shape)
#plot the images
fig = plt.figure(figsize=(10, 10))
for i in range(1, 11):
    ax = fig.add_subplot(1, 10, i)
    ax.imshow(train[i][0].cpu().detach().numpy().reshape(108,108), cmap='gray')
plt.show()

model = Model[args.model]()
model = model.to(mps_device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()

accs_base = []
# time the epoch
for epoch in range(args.nr_epochs):
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
    accs_base.append(acc)
    end = time.time()
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100), 
          "Time %.2f" % (end-start), "sec")

# save the model
if args.save_model:
    if not os.path.exists(os.path.join(args.save_path, args.model)):
        os.makedirs(os.path.join(args.save_path, args.model))
    # check which index to save the model to
    if os.path.exists(os.path.join(args.save_path, args.model, "model.pt")):
        i = 1
        while os.path.exists(os.path.join(args.save_path, args.model, "model%d.pt" % i)):
            i += 1
        torch.save(model.state_dict(), os.path.join(args.save_path, args.model, "model%d.pt" % i))
        # save the accs
        torch.save(accs_base, "accs_base_03_%d.pt" % i)