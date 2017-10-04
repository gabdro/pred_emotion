import pandas as pd
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Activation, Dropout, Flatten, Dense, Reshape, merge
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

import tensorflow as tf
config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="1", allow_growth=True))
tf.Session(config=config)


def build_model(shape):
    #Input and ResNet
    input_tensor = Input(shape=shape)
    resnet_model = ResNet50(include_top=False,
                            weights=None,
                            input_tensor=input_tensor)
    #output layers
    #dense = Flatten()(Dense(2048,activation='relu'))
    output_model = Sequential()
    output_model.add(Flatten(input_shape=resnet_model.output_shape[1:]))
    output_model.add(Dense(2048,activation='relu'))
    output_model.add(Dropout(0.5))
    output_model.add(Dense(7,activation='softmax'))
    
    #concat
    model = Model(inputs=resnet_model.input,
                  outputs=output_model(resnet_model.output))
    return model



def load_dataset(DATA_DIR,load_limit=None):
    row,col = 224,224
    xs = []
    ys = []
    for filename in glob.glob(DATA_DIR)[:load_limit]:
        img = Image.open(filename).resize((row,col))
        img = img.convert("RGB")
        img = np.array(img)/255.
        xs.append(img)
        
        # To:Do One-hotに変更した方が良いかも
        emotion = LABEL[filename.split("/")[-2]]
        ys.append(emotion)
        
    return np.array(xs),np.array(ys)



# load data
HEADER=['filename',
        'valence',
        'arousal',
        'anger',
        'disgust',
        'fear',
        'joy',
        'sadness',
        'surprise',
        'neutral']


LABEL={'anger':0,
       'disgust':1,
       'joy':2,
       'fear':3,
       'sadness':4,
       'surprise':5,
       'neutral':6
      }
iLABEL={}
for c,i in LABEL.items():
    iLABEL[i]=c
    
NUM_CLASSES=7

# To:Do now not using
label_data = pd.read_csv("dataset/Emotion6/ground_truth.txt",
                          delimiter='\t',
                          skiprows=1,
                          names=HEADER)
# parametter
#To:Do argparse
seed = 9
epochs = 100
batch_size=32

tb_cb = keras.callbacks.TensorBoard(log_dir="logs/",
                                    histogram_freq=1)
cp_cb = keras.callbacks.ModelCheckpoint(filepath="MODEL/mode_{epoch:02d}_vloss{val_loss:.2f}_vacc{val_acc:.2f}.hdf5",
                                        save_best_only=True)

# load dataset
Xs, Ys = load_dataset("dataset/Emotion6/images/*/*")
print(Xs.shape[1:])
X_train,X_test,Y_train,Y_test = train_test_split(Xs,Ys,test_size=0.2,random_state=seed)

model = build_model(X_train.shape[1:])
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

Y_train = keras.utils.to_categorical(Y_train,NUM_CLASSES)
Y_test = keras.utils.to_categorical(Y_test,NUM_CLASSES)
model.fit(X_train,Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test,Y_test),
          callbacks=[tb_cb,cp_cb])
