import torch
import torchvision
import torch.nn as nn
from random import randint

def transformations(pad_image=True, add_noise=True, noise_factor=0.5, max_padding=80):
    # transform the data:
    transforms = [torchvision.transforms.ToTensor()]
    if pad_image:
        # transforms.append(ParamLambda(pad_my_image, max_padding))
        transforms.append(torchvision.transforms.Lambda(pad_my_image))
    if add_noise:
        # transforms.append(ParamLambda(add_gausian_noise, noise_factor))
        transforms.append(torchvision.transforms.Lambda(add_gausian_noise))
    transforms.append(torchvision.transforms.Normalize((0.1307,), (0.3081,)))
    # transforms.append(torchvision.transforms.Lambda(lambda x: torch.stack((x[0],)*3, axis=0)))
    # standardize the data between 0 and 1
    # transforms.append(torchvision.transforms.Lambda(standardize))
    transform = torchvision.transforms.Compose(transforms)
    return transform

def pad_my_image(image):
    """
    Crop the images so only a specific region of interest is shown to my PyTorch model
    """
    max_padding = 80
    left = randint(0, max_padding)
    top = randint(0, max_padding)
    right = max_padding - left
    bottom = max_padding - top
    transform = torchvision.transforms.Pad([left, top, right, bottom])
    return transform(image)

def add_gausian_noise(image):
    """
    Add gausian noise to the image
    """
    # create noise tensor with same size as image
    factor = 0.8
    # random tensor between 0 and 1 with size of image
    noise = torch.randn(image.size()) * factor

    # # standardize the noise between 0 and 1
    # noise = (noise - noise.min()) / (noise.max() - noise.min())
    image = image + noise
    # standardize the image between 0 and 1
    # image = (image - image.min()) / (image.max() - image.min())
    return image

# generate concepts from images
concepts = ["small_circle", "big_circle", "circle_occur", "horiz_line", "vert_line", "diag_line", "hook_down", "hook_up"]
concept_dict = {0: [0, 1, 1, 0, 0, 0, 1, 1],
                1: [0, 0, 0, 1, 0, 1, 0, 0],
                2: [0, 0, 0, 1, 0, 1, 0, 1],
                3: [0, 0, 0, 0, 0, 0, 1, 1],
                4: [0, 0, 0, 1, 1, 1, 0, 0],
                5: [0, 0, 0, 1, 1, 0, 1, 0],
                6: [1, 0, 1, 0, 0, 0, 0, 1],
                7: [0, 0, 0, 1, 1, 0, 0, 0],
                8: [1, 0, 1, 0, 0, 0, 0, 0],
                9: [1, 0, 1, 0, 0, 0, 1, 0],}

class ConceptLoss(nn.Module):
    def __init__(self):
        super(ConceptLoss, self).__init__()

    def forward(self, output, target):
        # for each row do a cross entropy loss
        loss = 0
        for i in range(output.shape[0]):
            out, targ = output[i].to("mps"), target[i].to("mps")
            # mean squared error
            loss += torch.sum((out-targ)**2)
        return loss
