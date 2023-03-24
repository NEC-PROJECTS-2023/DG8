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
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint 
'''
def gistGaborFilters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F) #getting Gabor features
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def gistGaborFeatures(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

features_filter = gistGaborFilters()

X = []
Y = []
for root, dirs, directory in os.walk('images'):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j])
            img = cv2.resize(img, (64,64))
            img = gistGaborFeatures(img, features_filter)
            label = 0
            if name == 'non-influence':
                label = 1
            X.append(img)
            Y.append(label)
            print(root+" "+str(label))

X = np.asarray(X)
Y = np.asarray(Y)
print(Y)

np.save('model/X.txt',X)
np.save('model/Y.txt',Y)
'''
X = np.load('model/X.txt.npy')
Y = np.load('model/Y.txt.npy') 

X = X.astype('float32')
X = X/255
    
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
X_train, X_test1, y_train, y_test1 = train_test_split(X, Y, test_size=0.1) #split dataset into train and test

alex = Sequential()
alex.add(Convolution2D(32, 3, 3, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
alex.add(MaxPooling2D(pool_size = (2, 2)))
alex.add(Convolution2D(32, 3, 3, activation = 'relu'))
alex.add(MaxPooling2D(pool_size = (2, 2)))
alex.add(Flatten())
alex.add(Dense(output_dim = 256, activation = 'relu'))
alex.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
alex.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/alex_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/alex_weights.hdf5', verbose = 1, save_best_only = True)
    hist = alex.fit(X_train, y_train, batch_size=16, epochs=15, shuffle=True, verbose=1, validation_data=(X_test, y_test), callbacks=[model_check_point])
    f = open('model/alex_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
else:
    alex.load_weights("model/alex_weights.hdf5")
predict = alex.predict(X_test)
predict = np.argmax(predict, axis=1)
testY = np.argmax(y_test, axis=1)
a = accuracy_score(testY,predict)*100      
print(a)           

