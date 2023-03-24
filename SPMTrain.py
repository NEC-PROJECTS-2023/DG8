import numpy as np
import pandas as pd
import cv2
import os
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn.model_selection import train_test_split
from keras.applications import ResNet50
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
'''
def getSPMFeatures(img):
    return cv2.pyrDown(img)

X = []
Y = []
for root, dirs, directory in os.walk('images'):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j])
            img = cv2.resize(img, (64,64))
            img = getSPMFeatures(img)
            label = 0
            if name == 'non-influence':
                label = 1
            X.append(img)
            Y.append(label)
            print(root+" "+str(label))

X = np.asarray(X)
Y = np.asarray(Y)
print(Y)

np.save('model/spm_X.txt',X)
np.save('model/spm_Y.txt',Y)
'''
X = np.load('model/spm_X.txt.npy')
Y = np.load('model/spm_Y.txt.npy')

img = X[0]
img = cv2.resize(img, (200,200))
cv2.imshow("aa", img)
cv2.waitKey(0)

X = X.astype('float32')
X = X/255
    
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test

#defining RESNET object and then adding layers for imagenet with CNN and max pooling filter layers
resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
for layer in resnet.layers:
    layer.trainable = False
resnet_model = Sequential()
resnet_model.add(resnet)
resnet_model.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
resnet_model.add(MaxPooling2D(pool_size = (1, 1)))
resnet_model.add(Convolution2D(32, 1, 1, activation = 'relu'))
resnet_model.add(MaxPooling2D(pool_size = (1, 1)))
resnet_model.add(Flatten())
resnet_model.add(Dense(output_dim = 256, activation = 'relu'))
resnet_model.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
resnet_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/resnet_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/resnet_weights.hdf5', verbose = 1, save_best_only = True)
    hist = resnet_model.fit(X, Y, batch_size=16, epochs=15, shuffle=True, verbose=1, validation_data=(X_test, y_test), callbacks=[model_check_point])
    f = open('model/resnet_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
else:
    resnet_model.load_weights("model/resnet_weights.hdf5")
predict = resnet_model.predict(X_test)
predict = np.argmax(predict, axis=1)
testY = np.argmax(y_test, axis=1)
a = accuracy_score(testY,predict)*100      
print(a)     
           

