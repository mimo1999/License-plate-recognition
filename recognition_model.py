//char74k Dataset used for training
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
X_train=[]
X_test=[]
Y_train=[]
Y_test=[]
count = 0
#file_loc = 'D:/English/Fnt/Sample0'
file_loc = 'D:/labels/'
for i in range(0,36):
    if i<=9:
        #dir_path = file_loc + '0' + str(i)+'/'
        dir_path = file_loc + str(i)+ '/'
    else:
        #dir_path = file_loc + str(i)+'/'
        dir_path = file_loc + chr(55+i) +'/'
    for filename in os.listdir(dir_path):
        img = cv2.imread(os.path.join(dir_path, filename))
        if img is not None:
            temp=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            temp=cv2.resize(temp,(28, 28),interpolation=cv2.INTER_CUBIC)
            a, temp = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            bg = np.ones((32, 32))
            bg = bg*255
            bg[2:30, 2:30] = temp
            #temp = cv2.GaussianBlur(temp, (3, 3), 0)
            temp1=np.zeros(36)
            temp1[i]=1
            if count%3==0:
                X_test.append(bg)
                Y_test.append(temp1)
                X_train.append(bg)
                Y_train.append(temp1)
            else:
                X_train.append(bg)
                Y_train.append(temp1)
            count +=1
 
file_loc = 'D:/English/Bmp/Sample0'
mask_loc = 'D:/English/Msk/Sample0'
for i in range(1,37):
    if i<=9:
        dir_path = file_loc + '0' + str(i)+'/'
        msk_path = mask_loc + '0' + str(i)+'/'
    else:
        dir_path = file_loc + str(i)+'/'
        msk_path = mask_loc + str(i)+'/'
    for filename, mskname in zip(os.listdir(dir_path), os.listdir(msk_path)):
        img = cv2.imread(os.path.join(dir_path, filename))
        msk = cv2.imread(os.path.join(msk_path, mskname))
        if img is not None:
            temp=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
            outer = np.ones_like(temp)
            outer = outer * 255
            outer[mask == 255] = temp[mask == 255]
            temp=cv2.resize(outer,(32, 32),interpolation=cv2.INTER_CUBIC)
            temp = cv2.GaussianBlur(temp, (3, 3), 0)
            temp1=np.zeros(36)
            temp1[i-1]=1
            if count%20==0:
                X_test.append(temp)
                Y_test.append(temp1)
                X_train.append(temp)
                Y_train.append(temp1)
            else:
                X_train.append(temp)
                Y_train.append(temp1)
            count +=1

file_loc = 'D:/English/Fnt/Sample0'
for i in range(1,37):
    if i<=9:
        dir_path = file_loc + '0' + str(i)+'/'
    else:
        dir_path = file_loc + str(i)+'/'
    for filename in os.listdir(dir_path):
        img = cv2.imread(os.path.join(dir_path, filename))
        if img is not None:
            temp=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            temp=cv2.resize(temp,None, fx=0.25, fy=0.25,interpolation=cv2.INTER_CUBIC)
            temp = cv2.GaussianBlur(temp, (3, 3), 0)
            temp1=np.zeros(36)
            temp1[i-1]=1
            if count%20==0:
                X_test.append(temp)
                Y_test.append(temp1)
                X_train.append(temp)
                Y_train.append(temp1)
            else:
                X_train.append(temp)
                Y_train.append(temp1)
            count +=1
X_train=np.asarray(X_train)
X_test=np.asarray(X_test) 
Y_train=np.asarray(Y_train)
Y_test=np.asarray(Y_test)
print(X_train.shape)
print(X_test.shape)
print(count)
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 1)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
num_classes = 36
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))
epochs = 9
batch_size = 128

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('D:/English/my_label_model3.h5')
