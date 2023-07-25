import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from arguments import get_arguments
from utils import transformations
from GAA_model import GAA, PersistenceDiagrams

# detect if MPS is available
mps_device = torch.device("mps")

#get arguments from arguments.py
args = get_arguments()
print(args)





# define hyperparameters
batch_size = args.batch_size
num_epochs = args.nr_epochs
learning_rate = args.lr

# normalization used in MNIST dataset
transform = transformations()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# create PersistenceDiagrams instance
pd = PersistenceDiagrams(image_size=64, num_classes=10)

# create GAA model
model = GAA(image_size=64, num_classes=10).to(mps_device)

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train the model
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader, 0):
        # convert images to persistence diagrams
        attention_maps = model(images.to(mps_device))
        pd.generate_diagrams(images.numpy(), attention_maps.detach().cpu().numpy())
        diagrams = pd.get_prototypes()
        diagrams = torch.tensor(diagrams, dtype=torch.float32).to(mps_device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images.to(mps_device))
        loss = criterion(outputs, labels.to(mps_device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')
