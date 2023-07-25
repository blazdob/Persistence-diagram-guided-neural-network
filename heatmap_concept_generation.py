import cv2
import torch
import copy
import random
import torchvision

from utils import transformations, concept_dict
from arguments import get_arguments

args = get_arguments()
print(args)

drawing=False # true if mouse is pressed
mode=True # if True, draw rectangle. Press 'm' to toggle to curve 



# mouse callback function
def draw_heatmap(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode
    overlay = im.copy()
    alpha = 0.1
    output = im.copy()
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),40)
                cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
                current_former_x = former_x
                current_former_y = former_y
                #print former_x,former_y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),40)
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
            current_former_x = former_x
            current_former_y = former_y
    return former_x,former_y 

######## Transform the data ########
transform = transformations()

# load mnist dataset using dataloader and choose 20 images to draw heatmap on
train = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)

# get some random training images
# sample 10 random images from the validation set
num_annotated_images = 300
random.seed(1)
indices = random.sample(range(len(train)), num_annotated_images)
images = [train[i][0] for i in indices]
labels = [train[i][1] for i in indices]
images = torch.stack(images)

# save the list of images and labels
heatmap_images = []
concept_labels = []
# read the image using cv2
for i, (label, image) in enumerate(zip(labels, images)):
    # convert tensor to numpy array
    squeezed_image = image.squeeze()
    image = squeezed_image.numpy()
    # plot image
    # image = image.reshape(image.shape[1],image.shape[2], 3)

    concept_labels.append(concept_dict[label])
    # make a distinct copy of the image
    im = copy.deepcopy(image)

    # # # create cv image obejct from numpy array
    cv2.namedWindow("draw_heatmap")
    cv2.setMouseCallback('draw_heatmap',draw_heatmap)
    while(1):
        cv2.imshow('draw_heatmap',im)
        # print the matri that has been handdrawn

        k=cv2.waitKey(1)&0xFF
        if k==27:
            # save the image with the label and the original name
            # add to the list of heatmap images
            converted_heatmap = torch.from_numpy(im)
            heatmap_images.append(converted_heatmap)
            # print(sum(sum(im)), sum(sum(image)), sum(sum(squeezed_image)))
            # cv2.imwrite(f'data/anotated_MNIST/raw/image_{i}_target_{label}.png',image)
            # save tensor image with the label and the original name 
            # torch.save(squeezed_image, f'data/anotated_MNIST/raw/image_{i}_target_{label}.pt')
            # f'data/anotated_MNIST/heatmaps/im{i}_{label}_heatmap.png',squeezed_image)
            break
    cv2.destroyAllWindows()

torch.save(images, 'data/anotated_MNIST/raw/images_01.pt')
torch.save(labels, 'data/anotated_MNIST/raw/labels_01.pt')
# save the list of images and labels
torch.save(heatmap_images, 'data/anotated_MNIST/heatmaps/heatmaps_01.pt')
torch.save(concept_labels, 'data/anotated_MNIST/raw/concept_labels_01.pt')
