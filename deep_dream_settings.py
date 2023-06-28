IMAGE_PATH = './inp_1.jpg'
DEVICE = 'cpu'
SAVED_IMAGE_DIR = './saved_images/'

VARIATION_LOSS_WEIGHT = 0  # 0.000001
FEATURE_LOSS_WEIGHT = 1

INPUT_IMAGE_SIZE = 500
INPUT_IMAGE_SIZE_LIST = range(100, 650, 100)
FEATURE_MAP_IDX_TO_MAXIMIZE = 3 # Use 'None' to maximize all features
LEARNING_RATE = 0.01
NUM_EPOCHS = 50
ENABLE_IMAGE_GRAD_SMOOTHENING = True
LIST_OF_AVAILABLE_MODELS = ['vgg16', 'resnet50']
MODEL_NAME = 'vgg16'

GAUSSIAN_FILTER_SIGMA = 0.5  # 3

if __name__ == '__main__':
    from main import main

    main()
