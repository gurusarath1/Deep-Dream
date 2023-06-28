import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import deep_dream_utils
import deep_dream_settings
import utils

from deep_dream_models import Vgg16_truncated, Resnet50_truncated
from deep_dream_utils import get_image_tensor, get_feature_loss, calc_variation_loss


def main():
    print('Deep Dream Started !')

    DEVICE = utils.get_device()

    dd_image = get_image_tensor(deep_dream_settings.IMAGE_PATH,
                                device=DEVICE,
                                transform=None,
                                add_batch_dim=True)

    if deep_dream_settings.MODEL_NAME == 'vgg16':
        dd_model = Vgg16_truncated()
    elif deep_dream_settings.MODEL_NAME == 'resnet50':
        dd_model = Resnet50_truncated()
    # Keep the model in eval mode, since we are not training the model
    # We only need the feature maps from a trained model
    dd_model = dd_model.to(DEVICE).eval()

    target_image = dd_image

    image_shape = np.array(dd_image.shape)[2:]  # Remove batch and channel dim
    print(f'Image shape = {image_shape}')

    # Image size array -> scale the image linearly from 150 to 650
    image_size_array = deep_dream_settings.INPUT_IMAGE_SIZE_LIST
    for image_size in image_size_array:

        new_img_size = image_size

        print(f'size of img = {new_img_size}')

        resize_image_transform = transforms.Resize(new_img_size)

        # Image to optimize / Generated image / Target image
        target_image = Variable(resize_image_transform(target_image.detach().clone()),
                                requires_grad=True)  # Detach is needed so that there is no autograd relationship

        # We are only optimizing the generated image and not the model weights
        optimizer = torch.optim.Adam((target_image,), lr=deep_dream_settings.LEARNING_RATE)

        for epoch in (range(deep_dream_settings.NUM_EPOCHS)):

            try:
                optimizer.zero_grad()

                # Get the feature maps of the generated image by running it through the model
                target_image_feature_maps = dd_model(target_image)

                # Total loss = weighted sum of all the losses
                total_loss = -(
                        deep_dream_settings.FEATURE_LOSS_WEIGHT * get_feature_loss(target_image_feature_maps,
                                                                                   deep_dream_settings.FEATURE_MAP_IDX_TO_MAXIMIZE)) + \
                             (deep_dream_settings.VARIATION_LOSS_WEIGHT * calc_variation_loss(target_image)
                              )

                # Calculate gradients
                total_loss.backward()

                if deep_dream_settings.ENABLE_IMAGE_GRAD_SMOOTHENING:
                    # Smoothen the gradients for smooth image quality
                    image_grads = target_image.grad.clone().cpu().numpy()
                    image_grads = image_grads.squeeze(0)
                    blurred_grads = gaussian_filter(image_grads, deep_dream_settings.GAUSSIAN_FILTER_SIGMA)
                    # Update the new gradients manually
                    target_image.grad = torch.tensor(blurred_grads).unsqueeze(0).to(DEVICE)

                # Apply gradient descent once
                # This step updates the output image
                optimizer.step()

                if epoch % 10 == 0:
                    print(f'Eopch = {epoch}  Loss = {total_loss.item()}')
                    save_image = target_image.detach().cpu()
                    torchvision.utils.save_image(save_image,
                                                 deep_dream_settings.SAVED_IMAGE_DIR + f'op_image_{image_size}_{epoch}.png')

            # If the user presses ctrl-c while optimization is running save the output
            except KeyboardInterrupt:
                save_image = target_image.detach().cpu()
                torchvision.utils.save_image(save_image,
                                             deep_dream_settings.SAVED_IMAGE_DIR + 'op_image_' + str(epoch) + '.png')


if __name__ == '__main__':
    main()
