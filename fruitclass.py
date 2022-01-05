import pandas as pd
import numpy as np
import os
from glob import glob
import random
import matplotlib.pylab as plt
paths=[]
for name in glob('C:/data science/project/New folder/fruit-recognition/*',recursive=True):
    paths.append(name)

paths


img_paths=[]
extra_paths=[]
for i in paths:
    for name in glob(i+'/*'):
        if name.endswith('.png'):
            a=name.split('/')
            img_paths.append([name,a[-2]])
        else:
            extra_paths.append(name)
            
extra_paths


extra_imgs=[]
for i in extra_paths:
    if i=='C:/data science/project/New folder/fruit-recognition/Apple/Total Number of Apples':
        for name in glob(i+'/*'):
            if name.endswith('.png'):
                extra_imgs.append([name,'Apple'])
    elif i=='C:/data science/project/New folder/fruit-recognition/Guava/guava total final':
        for name in glob(i+'/*'):
            if name.endswith('.png'):
                extra_imgs.append([name,'Guava'])
    elif i=='C:/data science/project/New folder/fruit-recognition/Kiwi/Total Number of Kiwi fruit':
        for name in glob(i+'/*'):
            if name.endswith('.png'):
                extra_imgs.append([name,'Kiwi'])
                
##extra_imgs[5000]


img_final=img_paths+extra_imgs
len(img_final)

labels=[]
for i in range(len(img_final)):
    labels.append(img_final[i][1])
    img_final[i]=img_final[i][0]
    
print(set(labels))

img_path = pd.Series(img_final).astype(str)

labels=pd.Series(labels)
data = pd.concat([img_path,labels],axis=1)
data.sample(5)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,5))
sns.countplot(x=data[1])

from sklearn.model_selection import train_test_split
train_set , test_set = train_test_split(data,test_size=0.2,random_state=17)
train_set.shape,test_set.shape


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

pd.set_option('display.max_colwidth', None)
data.describe()
data.head()


train_gen = ImageDataGenerator(validation_split=0.1)
test_gen = ImageDataGenerator()

train_data = train_gen.flow_from_dataframe(
    dataframe = train_set,
    x_col = 0,
    y_col = 1,
    target_size = (227,227),
    color_mode = 'rgb',
    class_mode = 'categorical',
    shuffle = True,
    subset = 'training'
)

val_data = train_gen.flow_from_dataframe(
    dataframe = train_set,
    x_col = 0,
    y_col = 1,
    target_size = (227,227),
    color_mode = 'rgb',
    class_mode = 'categorical',
    shuffle = False,
    subset = 'validation'
)

test_data = test_gen.flow_from_dataframe(
    dataframe = test_set,
    x_col = 0,
    y_col = 1,
    target_size = (227,227),
    color_mode = 'rgb',
    class_mode = 'categorical',
    shuffle = False
)

data[0][0]
##from keras.models import Sequential
###from keras.layers import Dense
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),##mapping
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(15, activation='softmax') ###otput
])
model.compile(
    optimizer=tf.optimizers.Adam(lr=0.000001),
    loss='categorical_crossentropy',#######have to look at oit
    metrics=['accuracy','Recall']
)
model.summary()

# can do in another model using target size ,batch size and class model\\both train and test//

history = model.fit(train_data,epochs=40,validation_data=val_data)

import matplotlib.pyplot as plt
#plotting the Accuracy of test and training sets
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#plotting the loss of test and training sets
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


y_pred = model.predict(test_data)
y_pred = np.argmax(y_pred,axis=1)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(test_data.labels,y_pred))


classes=[i for i in range(15)]
con_mat_df = pd.DataFrame(confusion_matrix(test_data.labels,y_pred),
                     index = classes, 
                     columns = classes)
import seaborn as sns
figure = plt.figure(figsize=(15, 8))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.cool,fmt='d')
plt.tight_layout()
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

