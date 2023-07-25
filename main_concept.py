import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision


from model import Model
from arguments import get_arguments
from utils import transformations, concept_dict, ConceptLoss

# detect if MPS is available
mps_device = torch.device("mps")

# set environment variable to enable MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

#get arguments from arguments.py
args = get_arguments()
print(args)

# normalization used in MNIST dataset
transform = transformations()

#seed the random number generator
torch.manual_seed(args.seed)
train = torchvision.datasets.MNIST('data',train=True,download=True,transform=transform)
test = torchvision.datasets.MNIST('data',train=False,download=True,transform=transform)

# add concepts to train and test datasets
trainloader = torch.utils.data.DataLoader(train, shuffle=True,
                                          batch_size=args.batch_size)
testloader = torch.utils.data.DataLoader(test, shuffle=True,
                                         batch_size=args.batch_size)

model = Model["CoLeNet5"]()
model = model.to(mps_device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
loss_fn = ConceptLoss()
 
args.nr_epochs = 30
# time the epoch
for epoch in range(args.nr_epochs):
    # time the epoch
    start = time.time()
    model.train()
    for i, (X_batch, y_batch) in enumerate(trainloader):
        X_batch, y_batch = X_batch.to(mps_device), y_batch.to(mps_device)
        y_pred = model(X_batch)

        # map the y_pred and concepts using the concept_dict
        true_concepts = torch.tensor(np.asarray(list(map(lambda x: np.array(concept_dict[x.item()]), y_batch))))
        # pred_concepts = torch.tensor(list(map(lambda x: np.array(concept_dict[torch.argmax(x).item()]), y_pred)))
        # use different loss function to compare the concepts
        loss = loss_fn(y_pred, true_concepts)
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
        # map the y_pred and concepts using the concept_dict
        true_concepts = torch.tensor(np.array(list(map(lambda x: concept_dict[x.item()], y_batch)))).to("mps")
        # pred_concepts = np.array(list(map(lambda x: concept_dict[torch.argmax(x).item()], y_pred)))

        # round the y_pred to 0 or 1
        y_pred_round = torch.round(y_pred)
        # print(y_pred_round)
        # do a element-wise comparison between the true_concepts and y_pred_round
        y_pred_round
        # print(y_pred_round, true_concepts, y_pred_round == true_concepts)
        acc += torch.all((y_pred_round == true_concepts), 1).sum()
        count += len(y_batch)
    acc = acc / count
    end = time.time()
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100), 
          "Time %.2f" % (end-start), "sec")

# save the model
if args.save_model:
    if not os.path.exists(os.path.join(args.save_path, args.model)):
        os.makedirs(os.path.join(args.save_path, args.model))
    # check which index to save the model to
    if os.path.exists(os.path.join(args.save_path, args.model, "model_concept.pt")):
        i = 1
        while os.path.exists(os.path.join(args.save_path, args.model, "model_concept%d.pt" % i)):
            i += 1
        torch.save(model.state_dict(), os.path.join(args.save_path, args.model, "model_concept%d.pt" % i))
