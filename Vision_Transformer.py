'''
# Import

'''

import torch
import torch.nn as nn #layers, loss functions...
import torch.nn.functional as F
import torch.optim as optim #optimization method
import torchvision
from torch.utils.data import DataLoader #convert dataset into patches
from torchvision import datasets, transforms #to acess datasets and transformation to appply
import numpy as np
import random
import matplotlib.pyplot as plt #visualizing


'''
# Set up Seed

'''

# Set the seed so that in CPU, GPU nothing is going to be fully random and model behave the same throughout training
torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)

'''
## Checking to make sure using GPU
'''

import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

'''
# Set up Hyperparameters
'''

# Setting the hyperparameters
# All of these can be changed to experiment and tune the model

BATCH_SIZE = 128       # Number of images to process at once (higher = faster but uses more memory). Common to use powers of 2
EPOCHS = 10            # Number of full passes through the training dataset (entire dataset pass through = 1 epoch)
LEARNING_RATE = 3e-4   # Controls how much the model updates weights each step. Smaller = slower but more stable learning

PATCH_SIZE = 4         # Size of each image patch (4x4). Usually it would be 16x16 but since this image is pretty small we split it into 4x4

NUM_CLASSES = 10       # Output dimension of classifier. CIFAR-10 has 10 categories

IMAGE_SIZE = 32        # Width/Height of input image (CIFAR-10 = 32x32)
CHANNELS = 3           # Number of color channels (RGB = 3)

EMBED_DIM = 256        # Dimension of each patch embedding (token). Core size of all token vectors, control the dim throughout
NUM_HEADS = 8          # Number of attention heads. Splits each token into 8 parts for multi-head self-attention
DEPTH = 6              # Number of transformer encoder blocks (layers stacked in the model). Each layer is a full block that contain everything

MLP = 512              # Hidden dimension inside the MLP block in each transformer layer (usually 2–4× EMBED_DIM)
DROP_RATE = 0.1        # Dropout probability — randomly drops some activations to prevent overfitting


'''
# Define Image Transformation
'''


# Define Image Transformations
transform = transforms.Compose([ # Container chains multiple image transformation together
    transforms.ToTensor(), # Convert PIL images to Pytorch tensors so we can pass them to the model
    transforms.Normalize((0.5),(0.5)) # Normalize the images. Shift the value to [-1.0, 1.0]
    # this make the model more stable, faster to converge (is when it reach a decent good spot - not always the best )
    # help introduce negative number which will help with the gradient, activation steps,...
])


'''
# Getting datasets and view it
'''

# Getting a Dataset

train_dataset = datasets.CIFAR10(root = "data", # where the data save at
                                 train = True, # get the train or the test data
                                 download = True, # true to download
                                 transform = transform) # apply what transformation and use the one we create up there


test_dataset = datasets.CIFAR10(root = "data",
                                train = False,
                                download = True,
                                transform = transform)



train_dataset


test_dataset

'''
# Converting datasets to dataloader
- Right now the data is in form of PyTorch Datasets
- Dataloader will turn our data into batches (mini-batches)
#### Why?
- More computationally efficient, less work for the hardware since it might not be able to look (store in memory) at 50000 images in one hit. So we break it into 128 images at a time (batch_size = 128)
- Give neural network more chances to update its gradients per epoch
'''

train_loader = DataLoader(dataset = train_dataset,
                          batch_size = BATCH_SIZE,
                          shuffle = True)
test_loader = DataLoader(dataset = test_dataset,
                         batch_size = BATCH_SIZE,
                         shuffle = False) # not shuffle with the test data


# Check the dataloader
print(f"Data Loader:  { train_loader, test_loader}")
print(f"Length of train_loader: {len(train_loader)} batches of {train_loader.batch_size}")
print(f"Length of test_loader: {len(test_loader)} batches of {BATCH_SIZE}")

'''
# Building Vision Transformer Model from Scratch

'''

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__() # Initialize the class so we can use all features from nn.Module (PyTorch base class)
        self.patch_size = patch_size # Save the patch size (4x4 here). We will use this to split the image

        self.proj = nn.Conv2d(
            in_channels=in_channels,    # Number of input channels (CIFAR-10 = 3 for RGB)
            out_channels=embed_dim,     # Number of output channels (the embedding dimension for each patch vector)
            kernel_size=patch_size,     # Size of the patch (e.g., 4) → creates patches of 4x4
            stride=patch_size           # Move the kernel by patch size. so here each patch is 4x4 and we are moving 4 pixel at the time so there will not be any overlapping patches
        )
        # Use a convolution to split the image into non-overlapping patches
        # Each patch gets projected into a vector of size EMBED_DIM (this acts like linear projection for each patch)

        num_patches = (img_size // patch_size) ** 2 # Calculate how many patches per image. CIFAR-10: 32x32 / 4x4 = 8x8 = 64 patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # Create a special CLS token (learnable vector) that will represent the entire image at the end
        # 1x1x256 — 1 batch, 1 token, 256 features

        self.pos_embed = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))
        # Positional embedding: tell the model the position of each patch (including CLS token)
        # Shape = (1, 65, 256) → 64 patches + 1 CLS = 65 total positions

    def forward(self, x:torch.Tensor): # x is a batch of images and in this case 128 and have the shape (128, 3, 32, 32) - 128 images, 3 channels, 32H, 32W
        B = x.size(0) # this is taking out the first value in that order so 128

        x = self.proj(x) # apply linear projection onto this batch of images transform the shape into (128, 256, 8, 8) - 128 images, 1x256 vector to represent each 4x4 patch (token size), 8 patches across W, 8 patches across H -> 64 patches per image

        x = x.flatten(2).transpose(1,2)
        # Step 1: flatten(2) → merge last two dims (8x8) into one
        # Shape goes from (B, 256, 8, 8) → (B, 256, 64)
        # Now we have 64 patch tokens per image, but still in (feature, token) format

        # Step 2: transpose(1, 2) → swap channel and token dimension
        # Shape becomes (B, 64, 256)
        # This makes it: 64 tokens per image, each with a 256-dim embedding
        # Final shape is (batch_size, num_patches, embed_dim)
        # This is the required input shape for transformer encoders

        cls_token = self.cls_token.expand(B, -1, -1)
         # Copy this CLS token and assign to each image. - 1 means keep that number the same so we are increasing the first one into B which is 128
         # by doing expand you are not wasting any memory space because these are not real copies they still point to the same original cls token, just for views
         # later on when add these cls token into the image tensor then it will allocate its new memory but for now you will not waste any

        x = torch.cat((cls_token,x), dim = 1)
        # it take the shape of CLS token and x and concatenating them togethe along the position 1 in the dimension.
        # cls token has shape (128,1,256) x has (128,64,256) so this will make x become (128,65,256)
        # at this step the cls token vector is real and will allocate memory
        # we have to do the expand is because .cat() require all tensors to have the same batch dimension

        x = x + self.pos_embed # adding the position embedding value into x just normal math 1 to 1
        return x
