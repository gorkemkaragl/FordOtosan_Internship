import glob
import cv2
import torch
import numpy as np
from constant import *


def tensorize_image(image_path_list, output_shape, cuda=False):
    # Create empty list
    local_image_list = []

    # For each image
    for image_path in image_path_list:

        print(image_path)
        # Access and read image
        image = cv2.imread(image_path)

        # Resize the image according to defined shape
        image = cv2.resize(image, output_shape)

        # Change input structure according to pytorch input structure
        torchlike_image = torchlike_data(image)

        # Add into the list
        local_image_list.append(torchlike_image)

    # Convert from list structure to torch tensor

    image_array = np.array(local_image_list, dtype=np.float32)
    torch_image = torch.from_numpy(image_array).float()

    # If multiprocessing is chosen
    if cuda:
        torch_image = torch_image.cuda()

    return torch_image

def tensorize_mask(mask_path_list, output_shape, n_class, cuda=False):
    # Create empty list
    local_mask_list = []

    # For each masks
    for mask_path in mask_path_list:

        # Access and read mask
        mask = cv2.imread(mask_path, 0)
        
        mask = cv2.resize(mask, output_shape, interpolation = cv2.INTER_NEAREST)

        # Apply One-Hot Encoding to image
        mask = one_hot_encoder(mask, n_class)

        # Change input structure according to pytorch input structure
        torchlike_mask = torchlike_data(mask)


        local_mask_list.append(torchlike_mask)

    mask_array = np.array(local_mask_list, dtype=np.int64)
    torch_mask = torch.from_numpy(mask_array).float()
    if cuda:
        torch_mask = torch_mask.cuda()

    return torch_mask


def image_mask_check(image_path_list, mask_path_list):


    # Check list lengths
    if len(image_path_list) != len(mask_path_list):
        print("There are missing failes ! Images and masks folder should have some number of files.")
        return False

    return True

############################ TODO ################################
def torchlike_data(data):
    

    # Obtain channel value of the input
    n_channels = data.shape[2]

    # Create and empty image whose dimension is similar to input
    torchlike_data_output = np.empty((n_channels, data.shape[0], data.shape[1]))
    for ch in range(n_channels):
        torchlike_data_output[ch] = data[:,:,ch]

    return torchlike_data_output

def one_hot_encoder(data, n_class):
    global encoded_label
    
   
    encoded_label = np.zeros((data.shape[0], data.shape[1], n_class), dtype=np.int32)  # Updated dtype
   
    global unique_values
    unique_values = np.unique(data)

    for i, unique_value in enumerate(np.unique(data)):
        encoded_label[:, :, i][data == unique_value] = 1
    return encoded_label
    
############################ TODO END ################################





if __name__ == '__main__':

    # Access images
    image_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
    image_list.sort()


    # Access masks
    mask_list = glob.glob(os.path.join(MASK_DIR, '*'))
    mask_list.sort()


    # Check image-mask match
    if image_mask_check(image_list, mask_list):
       
       
       batch_image_list = image_list[:BATCH_SIZE]
       batch_image_tensor = tensorize_image(batch_image_list, (224, 224))

       print("For features:\ndtype is "+str(batch_image_tensor.dtype))
       print("Type is: "+str(type(batch_image_tensor)))
       print("The size should be ["+str(BATCH_SIZE)+", 3, "+str(HEIGHT)+" , "+str(WIDTH)+"]")
       print("Size is "+str(batch_image_tensor.shape)+"\n")

       batch_mask_list = mask_list[:BATCH_SIZE]
       batch_mask_tensor = tensorize_mask(batch_mask_list, (HEIGHT, WIDTH),2)
    
       print("For labels:\ndtype is"+str(batch_mask_tensor.dtype))
       print("Type is: "+str(type(batch_mask_tensor)))
       print("The size should be ["+str(BATCH_SIZE)+", 2, "+str(HEIGHT)+" , "+str(WIDTH)+"]")
       print("Size is "+str(batch_mask_tensor.shape))  