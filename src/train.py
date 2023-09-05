from new_model import FoInternNet
from preprocess import tensorize_image, tensorize_mask, image_mask_check
import os
import glob
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tqdm
import torch,gc
import cv2
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker







#PARAMETERS#
valid_size = 0.3
test_size  = 0.2
batch_size = 4
epochs = 20
cuda = False
input_shape = (224, 224)
n_classes = 2

SRC_DIR = os.getcwd()
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')


# PREPARE IMAGE AND MASK LISTS
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()

# DATA CHECK
image_mask_check(image_path_list, mask_path_list)

# SHUFFLE INDICES
indices = np.random.permutation(len(image_path_list))

# DEFINE TEST AND VALID INDICES
test_ind  = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)

# SLICE TEST DATASET FROM THE WHOLE DATASET
test_input_path_list = image_path_list[:test_ind]
test_label_path_list = mask_path_list[:test_ind]

# SLICE VALID DATASET FROM THE WHOLE DATASET
valid_input_path_list = image_path_list[test_ind:valid_ind]
valid_label_path_list = mask_path_list[test_ind:valid_ind]

# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list = image_path_list[valid_ind:]
train_label_path_list = mask_path_list[valid_ind:]

# DEFINE STEPS PER EPOCH
steps_per_epoch = len(train_input_path_list)//batch_size

# CALL MODEL
model = FoInternNet(input_size=input_shape, n_classes=2)

# DEFINE LOSS FUNCTION AND OPTIMIZER
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
if cuda:
    model = model.cuda()

val_loss_save=[]
run_loss_save=[]

# TRAINING THE NEURAL NETWORK
for epoch in range(epochs):
    running_loss = 0
    for ind in range(steps_per_epoch):
        batch_input_path_list = train_input_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_label_path_list = train_label_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_input = tensorize_image(batch_input_path_list, input_shape, cuda)
        batch_label = tensorize_mask(batch_label_path_list, input_shape, n_classes, cuda)

    
        optimizer.zero_grad()
        outputs = model(batch_input)
        loss = criterion(outputs, batch_label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        #print(ind)
        if ind == steps_per_epoch-1:
            run_loss_save.append(running_loss)
            print('training loss on epoch {}: {}'.format(epoch, running_loss))
            val_loss = 0
            for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
                batch_input = tensorize_image([valid_input_path], input_shape, cuda)
                batch_label = tensorize_mask([valid_label_path], input_shape, n_classes, cuda)
                outputs = model(batch_input)
                loss = criterion(outputs, batch_label)
                val_loss += loss
                break
            val_loss_save.append(val_loss)
            print('validation loss on epoch {}: {}'.format(epoch, val_loss))

    torch.save(model,'C:/Users/Gorkem/Desktop/fo-intern-project/src/model.pt') 
    print("Model Saved!")
    

def predict(test_input_path_list):

    for i in tqdm.tqdm(range(len(test_input_path_list))):
        batch_test = test_input_path_list[i:i+1]
        test_input = tensorize_image(batch_test, input_shape, cuda)
        outs = model(test_input)
        out=torch.argmax(outs,axis=1)
        out_cpu = out.cpu()
        outputs_list=out_cpu.detach().numpy()
        mask=np.squeeze(outputs_list,axis=0)
            
            
        img=cv2.imread(batch_test[0])
        mg=cv2.resize(img,(224,224))
        mask_ind   = mask == 1
        cpy_img  = mg.copy()
        mg[mask==0 ,:] = (255, 0, 125)
        opac_image=(mg/2+cpy_img/2).astype(np.uint8)
        predict_name=batch_test[0]
        predict_path=predict_name.replace('images', 'predicted_masked')
        cv2.imwrite(predict_path,opac_image.astype(np.uint8))

predict(test_input_path_list)


epoch_list = list(range(1, epochs + 1))
norm_validation = [float(i) / sum(val_loss_save) for i in val_loss_save]
norm_train = [float(i) / sum(run_loss_save) for i in run_loss_save]


norm_validation_numpy = torch.tensor(norm_validation).detach().cpu().numpy()
norm_train_numpy = torch.tensor(norm_train).detach().cpu().numpy()

plt.figure(figsize=(16, 8))


plt.subplot(221)
plt.plot(epoch_list, norm_train_numpy, color="red")
plt.title("Train Loss", fontsize=13)


plt.subplot(222)
plt.plot(epoch_list, norm_validation_numpy, color="blue")
plt.title("Validation Loss", fontsize=13)


plt.subplot(212)
plt.plot(epoch_list, norm_train_numpy, color="red", label="Train Loss")
plt.plot(epoch_list, norm_validation_numpy, color="blue", label="Validation Loss")
plt.title("Train and Validation Loss", fontsize=13)
plt.legend()  


plt.savefig('C:/Users/Gorkem/Desktop/fo-intern-project/src/model-grafik.png')
plt.show()
