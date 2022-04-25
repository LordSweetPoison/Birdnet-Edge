"""Uses NCE loss and simclr method to vectorize images to a latent space"""
from filelock import FileLock
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm, trange

from torchvision import transforms
from PIL import Image
import os
from models import convnext_base
import ray
import ray.tune as tune
from ray.tune.schedulers import ASHAScheduler

from PIL import Image

class CustomDataSet(Dataset):
    """loades images from root dir"""
    def __init__(self, root_dir, image_size = 224):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0, 1),
            transforms.Resize((image_size, image_size))
        ])
        self.all_images = os.listdir(root_dir)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.root_dir, self.total_images[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

# vmap to allow batching, increases parrelism 
def create_NCE_loss(positive_scale):
    """creates NCE loss and scales positive by positive scale"""
    # normalize positive scale to not impact loss 
    positive_scale /= (positive_scale + 1.)
    negative_scale = 1. - positive_scale

    @torch.vmap
    def NCE_loss(latent_space):
        """similarity loss, positive examples are drawn closer, negative are pushed apart
        args:
            latent_space: tensor shape: (transforms, batch, laten_dims)
        """
        # cosine similarity of positive pairs, easy. 
        pos_cossim = torch.sum(F.cosine_similarity(latent_space[0], latent_space[1], dim = -1)) 

        # scale pos_cossim 
        pos_cossim *= positive_scale

        # cosine similarity of negative pairs, hard
        # get indices pairs of negative examples, no repeats (ei no combos)
        indices = torch.tensor([*range(latent_space.shape[1])])
        neg_combos = torch.combinations(indices)

        # negative sims between row 1 and other row 1 latens
        neg_cossim_row1 = torch.sum(F.cosine_similarity(latent_space[0][neg_combos[:, 0]], latent_space[0][neg_combos[:, 1]], dim = -1))
        # negative sims between row 2 and other row 2 latens
        neg_cossim_row2 = torch.sum(F.cosine_similarity(latent_space[1][neg_combos[:, 0]], latent_space[1][neg_combos[:, 1]], dim = -1))
        # negative sims between row 1 and 2
        neg_cossim_inter = torch.sum(F.cosine_similarity(latent_space[1][neg_combos[:, 0]], latent_space[1][neg_combos[:, 1]], dim = -1))
        # sum of all sims
        neg_cossim = neg_cossim_row1 + neg_cossim_row2 + neg_cossim_inter

        # scale neg_cossim
        neg_cossim *= negative_scale
        
        # calculate nce 
        nce = - torch.log(torch.exp(pos_cossim) / (torch.exp(pos_cossim) + torch.exp(neg_cossim)))
        return nce    

    return NCE_loss

def process_images(images, model, device, transform_pipe, transforms = 4):
    """calculates the latent space given a list of images
    O(n^2) relative to transforms
    
    transforms: how many transforms to compare against each other, I recommend you keep this low"""
    # define batch size
    batch_size = images.shape[0]

    assert batch_size % transforms == 0, f'batch size must be divisable by transforms instead: batch size: {batch_size} transforms: {transforms}'

    # load images to device
    images = images.to(device)

    # apply transforms, images will be shape (2 * batchsize, ...)
    images = torch.concat([transform_pipe(images) for _ in range(2)], axis = 0)
    
    # run model, latent_spaces shape: (2 * batch_size, laten dims)
    latent_spaces = model(images)

    # desired shape: (batch_size/transforms, 2, transforms, laten_dims)
    # 0, 1, 2, 3, 4, 5, 6, 7, 8, ... => | 0, 1, 2, 3 | 4, 5, 6, 7 | 8, ...
    # 0, 1, 2, 3, 4, 5, 6, 7, 8, ... => | 0, 1, 2, 3 | 4, 5, 6, 7 | 8, ...

    # reshape to (2, batch_size, laten_dims)
    latent_spaces = torch.reshape(latent_spaces, (2, batch_size, -1))

    # reshape to (2, batch_size/transforms, transforms, laten_dims)
    latent_spaces = torch.reshape(latent_spaces, (2, batch_size // transforms, transforms, -1))

    # swap axises to (batch_size/transforms, 2, transforms, laten_dims)
    latent_spaces = torch.transpose(latent_spaces, 0, 1)

    return latent_spaces

def train(model, opt, train_loader, device, transform_pipe, loss_fn, scheduler = None):
    """trains a model to vectorize images"""

    # put model on device
    model.to(device)

    train_loss = 0

    for images, _ in train_loader:
        # process images 
        latent_spaces = process_images(images, model, device, transform_pipe)

        # calculate loss 
        losses = loss_fn(latent_spaces)
        loss = torch.sum(losses)
        loss_mean = torch.mean(losses)

        # run backwards pass and update model params 
        loss.backward()
        opt.step()

        # zero opt state 
        opt.zero_grad()

        # modify loss 
        train_loss += loss_mean.item()

    # scale loss to length of dataset
    train_loss /= len(train_loader)

    # the reason you take sum for batch and mean for loss is you want to see average loss for the 
    # epoch (stays same regardless of batch size). However, the gradient should be contribute to 
    # each instance evenly, so you sum the batch. 

    # step scheduler if it exists 
    if scheduler: scheduler.step()

    # return train loss for this epoch
    return train_loss

def train_birds(config):
    """does a distributed hyperparam search"""

    # define hyperparameters 
    lr = config['lr']
    positive_scale = config['positive_scale']
    momentum = config['momentum']
    l2 = config['l2']

    # define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees = 0, translate = (.1, .1), scale = (.9, 1.1)),
        transforms.RandomAdjustSharpness(3),
        transforms.RandomPerspective(distortion_scale = 0.4, p = .5),
        transforms.ColorJitter(brightness = .4, saturation = .4, hue = .1)
    ])

    batch_size = 64

    # load in data 
    with FileLock(os.path.expanduser("~/.data.lock")):
        dataset = CustomDataSet('/home/ubuntu/Birdnet-Edge/segments')
        train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, 
                                num_workers = 4, drop_last = True)

    # define device to use a gpu if it is avialable 
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # use a convnext base pretrained but that also has my custom VectNext head 
    model = convnext_base(pretrained = True).to(device)
    model.train()

    # define an optimizer 
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum, weight_decay = l2)

    # create the loss and scale positive samples 
    loss_fn = create_NCE_loss(positive_scale)

    # schedule the search
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    while True:
        train_loss = train(
            model = model, 
            opt = optimizer, 
            train_loader = train_loader, 
            transform_pipe = transform,
            loss_fn = loss_fn,
            scheduler = scheduler)

        # report loss to tune
        tune.report(train_loss = train_loss)

def hypsearch():
    # define config 
    config = {
        "l2": tune.loguniform(1e-6, 1e-2),
        "lr": tune.loguniform(1e-4, 1e-1),
        "momentum": tune.uniform(.1, .99),
        "positive_scale": tune.uniform(.5, 4.)
    }

    scheduler_ray = ASHAScheduler()
        
    results = tune.run(
        train_birds,
        resources_per_trial = {"cpu":4, "gpu": 1 if torch.cuda.is_available() else 0},
        config = config,
        metric = "train_loss",
        stop={
            "training_iteration": 1
        },
        mode = "min",
        num_samples = 5,
        scheduler = scheduler_ray,
    )

    print("Best config is:", results.best_config)

def main():
    """does a distributed hyperparam search"""

    # define hyperparameters 
    lr = None
    positive_scale = None
    momentum = None
    l2 = None

    epochs = 10

    # define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees = 0, translate = (.1, .1), scale = (.9, 1.1)),
        transforms.RandomAdjustSharpness(3),
        transforms.RandomPerspective(distortion_scale = 0.4, p = .5),
        transforms.ColorJitter(brightness = .4, saturation = .4, hue = .1)
    ])

    batch_size = 64

    # load in data 
    with FileLock(os.path.expanduser("~/.data.lock")):
        dataset = CustomDataSet('/home/ubuntu/Birdnet-Edge/segments')
        train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, 
                                num_workers = 4, drop_last = True)

    # define device to use a gpu if it is avialable 
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # use a convnext base pretrained but that also has my custom VectNext head 
    model = convnext_base(pretrained = True).to(device)
    model.train()

    # define an optimizer 
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum, weight_decay = l2)

    # create the loss and scale positive samples 
    loss_fn = create_NCE_loss(positive_scale)

    # schedule the search
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    for epoch in trange(epochs):
        loss = train(
            model = model, 
            opt = optimizer, 
            train_loader = train_loader, 
            transform_pipe = transform,
            loss_fn = loss_fn,
            scheduler = scheduler)

        print(f"Epoch {epoch + 1} / {epochs}: NCE Loss: {loss}")

        


if __name__ == "__main__":
    hypsearch()
    
