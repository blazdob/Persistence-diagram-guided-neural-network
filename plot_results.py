import os
import torch
import torchvision

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from model import Model
from arguments import get_arguments
from utils import transformations
#get arguments from arguments.py

args = get_arguments()
print(args)

transform = transformations()

valid = torchvision.datasets.MNIST('data',train=False,download=True,transform=transform)
validloader = torch.utils.data.DataLoader(valid, shuffle=True, batch_size=args.batch_size)


# detect if MPS is available
mps_device = torch.device("mps")


# load the model    
model = Model[args.model]()
model.load_state_dict(torch.load(os.path.join(args.save_path, args.model, 'model13.pt')))
model.to(mps_device)
# Validation stage
model.eval()
acc = 0
count = 0
for i, (X_batch, y_batch) in enumerate(validloader):
    X_batch, y_batch = X_batch.to(mps_device), y_batch.to(mps_device)
    y_pred = model(X_batch)
    acc += (torch.argmax(y_pred, 1) == y_batch).float().sum()
    count += len(y_batch)
acc = acc / count
print("Validation accuracy %.2f%%" % (acc*100))


# take random batch from validation set
valid_X, valid_y = next(iter(validloader))

# plot first 10 images
images = valid_X[:10]
labels = valid_y[:10]
y_pred = torch.argmax(model(images.to(mps_device)), 1)
print(y_pred)
# plot images
fig, axs = plt.subplots(2, 5, figsize=(15, 5))
for i in range(2):
    for j in range(5):
        axs[i, j].imshow(images[i*5+j].cpu().detach().numpy().reshape(108,108,1), cmap='gray')
plt.show()

# plot the images
fig, axs = plt.subplots(2, 5, figsize=(15, 5))
for i in range(2):
    for j in range(5):
        axs[i, j].imshow(images[i*5+j].cpu().detach().numpy().reshape(108,108,1), cmap='gray')
        axs[i, j].set_title("Predicted: %d Actual: %d" % (y_pred[i*5+j], labels[i*5+j]))
        axs[i, j].axis('off')
plt.show()


# # plot the ROC curve
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# print(y_pred)
# for i in range(10):
#     fpr[i], tpr[i], _ = roc_curve(labels, y_pred)
#     roc_auc[i] = auc(fpr[i], tpr[i])
# plt.figure()
# lw = 2
# for i in range(10):
#     plt.plot(fpr[i], tpr[i], lw=lw,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(i, roc_auc[i]))
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()