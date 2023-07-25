import matplotlib.pyplot as plt
import torch
import random
import torchvision

from arguments import get_arguments
from utils import transformations

#get arguments from arguments.py
args = get_arguments()
print(args)

transform = transformations()

#seed the random number generator
torch.manual_seed(args.seed)
train = torchvision.datasets.MNIST('data',train=True,download=True,transform=transform)
test = torchvision.datasets.MNIST('data',train=True,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(train, shuffle=True,
                                          batch_size=args.batch_size)
testloader = torch.utils.data.DataLoader(test, shuffle=True,
                                         batch_size=args.batch_size)

# plot the images

# obtain one batch of training images

# sample 10 random images from the validation set
random.seed(1)
indices = random.sample(range(len(train)), 10)
images = [train[i][0] for i in indices]
labels = [train[i][1] for i in indices]
images = torch.stack(images)

print(images[1].max(), images[1].min())
# plot the images
fig, axs = plt.subplots(2, 5, figsize=(15, 5))
for i in range(2):
    for j in range(5):
        axs[i, j].imshow(images[i*5+j].cpu().detach().numpy().reshape(108,108,1), cmap='gray')
        axs[i, j].set_title("Actual: %d" % (labels[i*5+j]))
        axs[i, j].axis('off')
plt.show()