"""# Load libraries"""

import os, shutil
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

"""# Check files"""

path = "/C1-P1_Train"
class_names =  ["A","B","C"]
classNum = len(class_names)

for className in class_names:
    dir = path+"/"+className
    files = os.listdir(dir)
    imageNum = len(files)
    print(className + " : " , imageNum)

"""# Load Data"""

target_size = (224,224)
batch_size = 20

#ImageDataGenerator() 可以做一些影像處理的動作 
datagen = ImageDataGenerator(
    rescale = 1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.2,1.0],
    fill_mode='nearest',
    validation_split=0.2)

#以 batch 的方式讀取資料
train_batches = datagen.flow_from_directory(
        path,  
        target_size = target_size,  
        batch_size = batch_size,
        classes = class_names,
        subset='training')

valid_batches = datagen.flow_from_directory(
        path,
        target_size = target_size,
        batch_size = batch_size,
        classes = class_names,
        subset='validation')

"""# Build model"""

from tensorflow.python.keras.applications import resnet

# 凍結網路層數
FREEZE_LAYERS = 2
net = resnet.ResNet152(include_top=False, 
               weights="imagenet", 
               input_tensor=None,
               input_shape=(target_size[0],target_size[1],classNum),
               classes=classNum)
x = net.output
x = Flatten()(x)

# 增加 Dense layer，以 softmax 產生個類別的機率值
# x = Dense(256, activation='softmax', name='output2_layer')(x)

# 增加 DropOut layer
x = Dropout(0.5)(x)
output_layer = Dense(classNum, activation='softmax', name='softmax')(x)

# 設定凍結與要進行訓練的網路層
net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True

# 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
net_final.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# 輸出整個網路結構
# print(net_final.summary())

"""# Train"""

# 訓練模型
history = net_final.fit(train_batches,
                        steps_per_epoch = train_batches.samples // batch_size,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // batch_size,
                        epochs = 30)

net_final.save("models/mango_resnet152.h5")

