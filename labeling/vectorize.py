"""Uses NCE loss and simclr method to vectorize images to a latent space"""
import torch 
import torch.nn as nn
import torch.nn.functional as F

# vmap to allow batching, increases parrelism 
@torch.vmap
def NCE_loss(latent_space):
    """similarity loss, positive examples are drawn closer, negative are pushed apart
    args:
        latent_space: tensor shape: (transforms, batch, laten_dims)
        device: device to put these tensors on
    """
    # cosine similarity of positive pairs, easy. 
    pos_cossim = torch.sum(F.cosine_similarity(latent_space[0], latent_space[1], dim = -1))

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
    
    # calculate nce 
    nce = - torch.log(torch.exp(pos_cossim) / (torch.exp(pos_cossim) + torch.exp(neg_cossim)))
    return nce    

TRANSFORMS = 4
def process_images(images, model, device, transform_pipe):
    """calculates the latent space given a list of images"""
    # define batch size
    batch_size = images.shape[0]

    # load images to device
    images = images.to(device)

    # apply transforms, images will be shape (2 * batchsize, ...)
    images = torch.concat([transform_pipe(images) for _ in range(2)], axis = 0)
    
    # run model, latent_spaces shape: (2 * batch_size, laten dims)
    latent_spaces = model(images)

    # desired shape: (batch_size/TRANSFORMS, 2, TRANSFORMS, laten_dims)
    # 0, 1, 2, 3, 4, 5, 6, 7, 8, ... => | 0, 1, 2, 3 | 4, 5, 6, 7 | 8, ...
    # 0, 1, 2, 3, 4, 5, 6, 7, 8, ... => | 0, 1, 2, 3 | 4, 5, 6, 7 | 8, ...

    # reshape to (2, batch_size, laten_dims)
    latent_spaces = torch.reshape(latent_spaces, (2, batch_size, -1))

    # reshape to (2, batch_size/TRANSFORMS, TRANSFORMS, laten_dims)
    latent_spaces = torch.reshape(latent_spaces, (2, batch_size // TRANSFORMS, TRANSFORMS, -1))

    # swap axises to (batch_size/TRANSFORMS, 2, TRANSFORMS, laten_dims)
    latent_spaces = torch.transpose(latent_spaces, 0, 1)

    return latent_spaces


def train(model, opt, train_loader, device, transform_pipe, scheduler = None):
    """trains vectorizor"""

    # put model on device
    model.to(device)

    train_loss = 0

    for images, _ in train_loader:
        # process images 
        latent_spaces = process_images(images, model, device, transform_pipe)

        # calculate loss 
        losses = NCE_loss(latent_spaces)
        loss = torch.sum(losses)
        loss_mean = torch.mean(losses)

        # run backwards pass and update model params 
        loss.backward()
        opt.step()

        # zero opt state 
        opt.zero_grad()

        # modify loss 
        train_loss += loss_mean.item()

    # step scheduler if it exists 
    if scheduler: scheduler.step()

    # return train loss for this epoch
    return train_loss

if __name__ == "__main__":
    from models import convnext_base

    model = convnext_base()
    model.train()
    # https://aws.amazon.com/blogs/machine-learning/announcing-the-amazon-s3-plugin-for-pytorch/
    

