import torch.nn as nn
from utils import transformations
from arguments import get_arguments
import torch
import torchvision

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 784)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(784, 10)
        
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.layer2(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.dropout = nn.Dropout(0.2)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(13*351, 128)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.relu1(self.conv(x))
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = self.dropout(x)
        # print(x.shape)
        x = self.flat(x)
        # print(x.shape)
        x = self.fc(x)
        # print(x.shape)
        x = self.relu2(x)
        # print(x.shape)
        x = self.output(x)
        return x

class LeNet5(nn.Module):
    def __init__(self, channels=1):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, 6, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
 
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.act2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
 
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.act3 = nn.Tanh()
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(1*1*12000, 84)
        self.act4 = nn.Tanh()
        self.fc2 = nn.Linear(84, 10)
        
    def forward(self, x):
        # input 1x108x108, output 6x54x54
        x = self.act1(self.conv1(x))
        # print(x.shape)
        # input 6x54x54, output 6x27x27
        x = self.pool1(x)
        # print(x.shape)
        # input 6x27x27, output 16x23x23
        x = self.act2(self.conv2(x))
        # print(x.shape)
        # input 16x23x23, output 16x11x11
        x = self.pool2(x)
        # print(x.shape)
        # input 16x11x11, output 120x7x7
        x = self.act3(self.conv3(x))
        # print(x.shape)
        # input 120x7x7, output 120x3x3
        x  = self.pool3(x)
        # print(x.shape)
        # input 120x3x3, output 120
        x = self.flat(x)
        # print(x.shape)
        # input 120, output 84
        x = self.act4(self.fc1(x))
        # print(x.shape)
        # input 84, output 10
        x = self.fc2(x)
        # print(x.shape)
        return x

class CoLeNet5(nn.Module):
    def __init__(self, channels=1):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, 6, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
 
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.act2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
 
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.act3 = nn.Tanh()
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
 
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(1*1*12000, 84)
        self.act4 = nn.Tanh()

        self.fc2 = nn.Linear(84, 8)


        
    def forward(self, x):
        # input 1x108x108, output 6x54x54
        x = self.act1(self.conv1(x))
        # print(x.shape)
        # input 6x54x54, output 6x27x27
        x = self.pool1(x)
        # print(x.shape)
        # input 6x27x27, output 16x23x23
        x = self.act2(self.conv2(x))
        # print(x.shape)
        # input 16x23x23, output 16x11x11
        x = self.pool2(x)
        # print(x.shape)
        # input 16x11x11, output 120x7x7
        x = self.act3(self.conv3(x))
        # print(x.shape)
        # input 120x7x7, output 120x3x3
        x  = self.pool3(x)
        # print(x.shape)
        # input 120x3x3, output 120
        x = self.flat(x)
        # print(x.shape)
        # input 120, output 84
        x = self.act4(self.fc1(x))
        # print(x.shape)
        # input 84, output 10
        x = self.fc2(x)
        # print(x.shape)
        return x

class ConSim(nn.Module):
    def __init__(self, input_shape=680, output_shape=10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(680, 1024)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 256)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(256, 128)
        self.act4 = nn.ReLU()
        self.fc5 = nn.Linear(128, 64)
        self.act5 = nn.ReLU()
        self.fc6 = nn.Linear(64, 32)
        self.act6 = nn.ReLU()
        self.fc7 = nn.Linear(32, 10)


    def forward(self, x):
        # forwards pass
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.act4(self.fc4(x))
        x = self.act5(self.fc5(x))
        x = self.act6(self.fc6(x))
        x = self.fc7(x)
        return x
    
Model = dict(Baseline=Baseline,
             CNN=CNN,
             LeNet5=LeNet5,
             CoLeNet5=CoLeNet5
            )


if __name__ == "__main__":
    from torchviz import make_dot
    args = get_arguments()
    print(args)
    model = Model[args.model]()
    #get arguments from arguments.py

    # normalization used in MNIST dataset
    transform = transformations()

    #seed the random number generator
    torch.manual_seed(args.seed)
    train = torchvision.datasets.MNIST('data',train=True,download=True,transform=transform)
    test = torchvision.datasets.MNIST('data',train=False,download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(train, shuffle=True,
                                            batch_size=args.batch_size)
    testloader = torch.utils.data.DataLoader(test, shuffle=True,
                                            batch_size=args.batch_size)
    batch = next(iter(trainloader))
    for i, (X_batch, y_batch) in enumerate(trainloader):
        if i == 0:
            batch = X_batch#.to(args.device)
            break
    yhat = model(batch)
    make_dot(yhat, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")