import torchvision
import torch
import matplotlib.pyplot as plt
import utils
import deep_dream_settings

from deep_dream_settings import DEVICE
def get_image_tensor(image_path, device=DEVICE, transform=None, add_batch_dim=False, batch_dim_index=0):
    image = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB).to(device)
    # [channels, height, width].

    image = image / 255.0

    if add_batch_dim:
        image = torch.unsqueeze(image, batch_dim_index)
        # [batch_size, channels, height, width]

    if transform is not None:
        image = transform(image)

    print(image_path, '- ', image.shape, device)

    return image


# Reference: https://stackoverflow.com/questions/53623472/how-do-i-display-a-single-image-in-pytorch
def display_image(tensor_image, batch_dim_exist=False, batch_dim_index=0, save_image=False, file_name='saved_img.png'):
    if batch_dim_exist:
        plt.imshow(tensor_image.squeeze(dim=batch_dim_index).permute(1, 2,
                                                                     0))  # remove batch dim and Make the Channel dim last
    else:
        plt.imshow(tensor_image.permute(1, 2, 0))  # Make the Channel dim last

    if save_image:
        plt.savefig(deep_dream_settings.SAVED_IMAGE_DIR + file_name, bbox_inches='tight')
    else:
        plt.show()


def get_feature_loss(features: list, specific_feature_idx=None):
    loss = 0

    if specific_feature_idx is not None:
        loss = torch.mean(features[specific_feature_idx])
        return loss

    for f in features:
        loss += torch.mean(f)

    return loss / len(features)


def calc_variation_loss(generated_image):
    return torch.sum(torch.abs(generated_image[:, :, :, :-1] - generated_image[:, :, :, 1:])) + \
        torch.sum(torch.abs(generated_image[:, :, :-1, :] - generated_image[:, :, 1:, :]))