import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gudhi as gd

# define the class that encodes the images and calculates the persistence diagram prototypes
class PersistenceDiagrams:
    def __init__(self, image_size, num_classes):
        self.image_size = image_size
        self.num_classes = num_classes
        self.prototypes = []

    def generate_diagrams(self, images, attention_maps):
        # convert images to grayscale and resize to 64x64
        gray_images = np.mean(images, axis=-1)
        gray_images = np.expand_dims(gray_images, axis=-1)
        gray_images = np.repeat(gray_images, 3, axis=-1)
        # apply attention maps to images
        masked_images = np.multiply(gray_images, attention_maps)

        # compute Delaunay triangulation and persistence diagrams for each class
        for i in range(self.num_classes):
            indices = np.where(attention_maps[i] > 0)
            delaunay = gd.AlphaComplex(points=indices).create_simplex_tree()
            persistence = delaunay.persistence()
            diagrams = gd.plot_persistence_diagram(persistence)
            self.prototypes.append(np.mean(diagrams, axis=0))

    def get_prototypes(self):
        return self.prototypes

# define the neural network with PartialPrototypeLayer
class PDGNN(nn.Module):
    def __init__(self, image_size, num_classes):
        super(PDGNN, self).__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128*11*11, 512)
        self.fc2 = nn.Linear(512, num_classes + 10)
        self.proto_layer = PartialPrototypeLayer(num_classes + 10, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 128*11*11)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.proto_layer(x)
        return x

# define the PartialPrototypeLayer
class PartialPrototypeLayer(nn.Module):
    def __init__(self, num_classes, num_prototypes):
        super(PartialPrototypeLayer, self).__init__()
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.prototypes = nn.Parameter(torch.randn(num_classes, num_prototypes))

    def forward(self, x):
        # split the input tensor into two parts
        x1, x2 = torch.split(x, [x.shape[1] - self.num_prototypes, self.num_prototypes], dim=1)

        # calculate the L2 distance between each prototype and the second part of the input tensor
        distance = torch.cdist(self.prototypes, x2, p=2)

        # take the minimum distance as the similarity score for each class based on the bottleneck layer
        similarity = torch.min(distance, dim=1)[0]

        # concatenate the similarity scores with the first part of the input tensor
        out = torch.cat([x1, similarity.unsqueeze(1)], dim=1)
        return out