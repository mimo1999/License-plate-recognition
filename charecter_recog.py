import numpy as np
import cv2
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import random
import keras.backend as K
K.set_image_data_format('channels_last')
import keras
import os
import matplotlib.pyplot as plt
# %matplotlib inline
from keras.models import load_model
model = load_model(r"D:/English/my_model4.h5")
#model = load_model(r"C:\Users\KIIT\Documents\my_model.h5")
val = 'A'
lis = []
for i in range(0, 36):
    if i<10:
        lis.append(str(i))
    else:
        lis.append(val)
        val = chr(ord(val)+1)
print(lis)

file_name = 'Hyundai-i20-529993c.jpg_0341_0121_0412_0182_0046'
dir_name = 'D:/out/'
dir_path = dir_name + file_name + '/'
try_img = []
to_be_saved = []
i = 0
lst = sorted(os.listdir(dir_path))
#lst.sort()
for filename in lst:
        img = cv2.imread(os.path.join(dir_path, filename))
        if img is not None:
            bg = np.ones((32, 32))
            temp=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            to_be_saved.append(img)
            temp=cv2.resize(temp,(32, 32), interpolation=cv2.INTER_CUBIC)
            #temp = cv2.GaussianBlur(temp,(3, 3),0)
            kernel = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])
            #kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            #temp= cv2.filter2D(temp, -1, kernel)
            
            #a, temp = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            #temp = cv2.adaptiveThreshold(temp,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,4)
            #temp = cv2.dilate(temp, kernel=(3, 3), iterations=2)
            #temp = cv2.erode(temp, kernel=(1, 2), iterations=1)
            bg = bg*255
            #bg[6:26, 6:26] = temp
            #bg[2:30, 2:30] = temp
            bg = temp
            #bg = cv2.GaussianBlur(bg,(3, 3),0)
            try_img.append(bg)
try_img=np.asarray(try_img)
plt.imshow(try_img[2])
plt.show()
try_img = try_img.reshape(try_img.shape[0], 32, 32, 1)
try_img = try_img.astype('float32')
try_img /= 255
num_classes = 36
pred = model.predict_classes(try_img)
print(pred)
name_lis = []
for i in pred:
    print(lis[i], end=' ')
    name_lis.append(lis[i])
    
name = ' '.join(str(x) for x in name_lis)
print(name)
img_dir = 'D:/inputs/solved/'
img_path = img_dir+file_name+'.png'
fin = cv2.imread(img_path)
plt.imshow(fin)
plt.show()
