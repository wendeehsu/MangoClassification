"""# Load libraries"""

import os, shutil
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Activation, Dense, GlobalAveragePooling2D, Dropout
from keras.layers.core import Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf

"""# Check files"""

path = "./C1-P1_Train/"
class_names =  ["A","B","C"]
classNum = len(class_names)

"""# Load Data"""
traindf=pd.read_csv("train.csv", header=None)
traindf = traindf.rename(columns={0: "name", 1: "class"})
print(traindf.head())
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
train_batches = datagen.flow_from_dataframe(
        dataframe=traindf,
        directory=path,
        x_col="name",
        y_col="class",
        target_size = target_size,  
        batch_size = batch_size,
        subset='training')

valid_batches = datagen.flow_from_dataframe(
        dataframe=traindf,
        directory=path,
        x_col="name",
        y_col="class",
        target_size = target_size,
        batch_size = batch_size,
        subset='validation')

"""# Build model"""

net = InceptionV3(include_top=False, weights="imagenet")
x = net.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output_layer = Dense(class_num, activation='softmax')(x)

FREEZE_LAYERS = 2
# 設定凍結與要進行訓練的網路層
net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True

# 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
net_final.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
history = net_final.fit_generator(train_batches,
                                steps_per_epoch = train_batches.samples // batch_size,
                                validation_data=valid_batches,
                                validation_steps = valid_batches.samples // batch_size,
                                epochs=30)

net_final.save("models/mango_Incept.h5")

STEP_SIZE_VALID = valid_batches.n // valid_batches.batch_size
result = net_final.evaluate_generator(generator=valid_batches, steps=STEP_SIZE_VALID, verbose=1)
print("result = ", result)

# plot metrics
plt.plot(history.history['accuracy'])
plt.show()
plt.savefig('accuracy.jpg')

