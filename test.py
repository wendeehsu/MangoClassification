import os, shutil
import random
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

path = "/C1-P1_Dev"
class_names =  ["A","B","C"]

dic = {}
for className in class_names:
    dir = path+"/"+className
    files = os.listdir(dir)
    imageNum = len(files)
    randomNums = random.sample(range(imageNum), imageNum)
    dic[className] = imageNum

plt.bar(range(len(dic)), list(dic.values()), align='center')
plt.xticks(range(len(dic)), list(dic.keys()))
print(dic)
plt.show()

target_size = (224,224)
batch_size = 1

#ImageDataGenerator() 可以做一些影像處理的動作 
datagen = ImageDataGenerator(rescale = 1./255,)

#以 batch 的方式讀取資料
predict_batches = datagen.flow_from_directory(
        path,
        shuffle=False,
        target_size = target_size,  
        batch_size = batch_size,
        classes = class_names)

resnet = load_model("models/mango_resnet152.h5")
# print(resnet.summary())
filenames = predict_batches.filenames
nb_samples = len(filenames)
predict = resnet.predict(predict_batches, steps = nb_samples, verbose = 1)
y_pred = np.argmax(predict, axis=1)
print('confusion matrix')
print(confusion_matrix(predict_batches.classes, y_pred))
print(classification_report(predict_batches.classes, y_pred, target_names=class_names))
