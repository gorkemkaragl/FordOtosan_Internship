import os


JSON_DIR = '../data/jsons'


MASK_DIR  = '../data/masks'
if not os.path.exists(MASK_DIR):
    os.mkdir(MASK_DIR)


IMAGE_OUT_DIR = '../data/masked_images'
if not os.path.exists(IMAGE_OUT_DIR):
    os.mkdir(IMAGE_OUT_DIR)

IMAGE_DIR = '../data/images'


VISUALIZE = True

BACTH_SIZE = 4

HEIGHT = 224
WIDTH = 224

N_CLASS= 2