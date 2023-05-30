import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np

import deep_dream_utils
import deep_dream_settings
import utils

from deep_dream_models import Vgg16_truncated
from deep_dream_utils import get_image_tensor, get_feature_loss, calc_variation_loss

if __name__ == '__main__':
    print('Deep Dream Started !')

    DEVICE = utils.get_device()

    t = transforms.Compose(
        [
            # It was never required to normalize the input image
            # transforms.Normalize(NST_project_settings.IMAGE_MEANS, NST_project_settings.IMAGE_STDS),
            #transforms.Resize(deep_dream_settings.INPUT_IMAGE_SIZE),
        ]
    )

    dd_image = get_image_tensor(deep_dream_settings.IMAGE_PATH,
                                device=DEVICE,
                                transform=t,
                                add_batch_dim=True)

    dd_model = Vgg16_truncated().to(DEVICE).eval()  # Keep the model in eval mode, since we are not training the model
    # We only need the feature maps from a trained model

    target_image = dd_image

    image_shape = np.array(dd_image.shape)[2:] # Remove batch and channel dim
    print(f'Image shape = {image_shape}')

    OCTAVE_SCALE = 0.3

    i = 0
    image_size_array = [1000, 900, 800, 700, 600, 500]

    for n in range(-2, 3):

        new_img_size = image_size_array[i] #(image_shape * (OCTAVE_SCALE**n)).astype(int).tolist()
        i += 1

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
                total_loss = -(deep_dream_settings.FEATURE_LOSS_WEIGHT * get_feature_loss(target_image_feature_maps)) + \
                             (deep_dream_settings.VARIATION_LOSS_WEIGHT * calc_variation_loss(target_image))

                # Calculate gradients
                total_loss.backward()

                # Apply gradient descent once
                optimizer.step()

                if epoch % 10 == 0:
                    print(f'Eopch = {epoch}  Loss = {total_loss.item()}')
                    save_image = target_image.detach().cpu()
                    torchvision.utils.save_image(save_image,
                                                 deep_dream_settings.SAVED_IMAGE_DIR + 'op_image_' + str(
                                                     epoch) + '.png')

            # If the user presses ctrl-c while optimization is running save the output
            except KeyboardInterrupt:
                save_image = target_image.detach().cpu()
                torchvision.utils.save_image(save_image,
                                             deep_dream_settings.SAVED_IMAGE_DIR + 'op_image_' + str(epoch) + '.png')
